import os
import warnings
import sys
import json
from datetime import datetime
from pathlib import Path
import io

warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

try:
    import importlib.metadata
    if not hasattr(importlib.metadata, 'packages_distributions'):
        import importlib_metadata
        importlib.metadata.packages_distributions = importlib_metadata.packages_distributions
except (ImportError, AttributeError):
    pass

_original_stderr = sys.stderr
_suppress_stderr = io.StringIO()

def _suppress_importlib_error():
    sys.stderr = _suppress_stderr

def _restore_stderr():
    sys.stderr = _original_stderr

_suppress_importlib_error()

import requests
from typing import TypedDict, List, Dict, Any
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

_restore_stderr()

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable is required. Set it in your .env file or environment.")

LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "gpt-4o-mini")

SERVERS: Dict[str, str] = {
    "compliance": os.getenv(
        "COMPLIANCE_SERVER_URL",
        "https://jani-36-compliance-server.hf.space/retrieve"
    ),
    "risk": os.getenv(
        "RISK_SERVER_URL",
        "https://jani-36-risk-server.hf.space/retrieve"
    ),
    "internal": os.getenv(
        "INTERNAL_SERVER_URL",
        "https://jani-36-internal.hf.space/retrieve"
    ),
}

REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "120"))

class State(TypedDict, total=False):
    question: str
    expanded_query: str
    doc_types: List[str]
    raw_context: List[Dict[str, Any]]
    validated_context: List[Dict[str, Any]]
    answer: str
    explanation: str
    sources: List[Dict[str, Any]]
    audit_trail: str
    confidence: float
    execution_path: List[str]

llm = ChatOpenAI(
    model=LLM_MODEL_NAME,
    temperature=0.2,
    openai_api_key=OPENAI_API_KEY,
)

def analyze_query(state: State):
    question = state["question"]
    path = state.get("execution_path", [])
    path.append("analyzer")

    # Fetch document summaries from all servers
    summaries_by_type = {}
    for doc_type, base_url in SERVERS.items():
        try:
            # Construct summaries URL (replace /retrieve with /summaries)
            summaries_url = base_url.replace("/retrieve", "/summaries")
            response = requests.get(summaries_url, timeout=REQUEST_TIMEOUT)
            if response.status_code == 200:
                data = response.json()
                summaries_by_type[doc_type] = data.get("summaries", [])
            else:
                summaries_by_type[doc_type] = []
        except Exception as e:
            print(f"Warning: Could not fetch summaries from {doc_type} server: {e}")
            summaries_by_type[doc_type] = []

    # Build summaries text for the prompt
    summaries_text = ""
    for doc_type in ["compliance", "risk", "internal"]:
        summaries = summaries_by_type.get(doc_type, [])
        if summaries:
            summaries_text += f"\n\nAvailable documents in {doc_type}:\n"
            for summary_item in summaries:
                doc_name = summary_item.get("doc_name", "unknown")
                summary = summary_item.get("summary", "")
                if summary:
                    summaries_text += f"- {doc_name}: {summary[:500]}...\n" if len(summary) > 500 else f"- {doc_name}: {summary}\n"
        else:
            summaries_text += f"\n\nAvailable documents in {doc_type}: (no documents indexed yet)\n"

    system = SystemMessage(
        content=(
            "You are a routing classifier for a compliance chatbot.\n"
            "Available doc types: compliance, risk, internal.\n"
            "Use the document summaries below to determine which doc types are most relevant to the user's question.\n"
            "Consider the content and topics covered in each document when making your routing decision.\n"
            f"{summaries_text}\n"
            "Return 1-3 types, comma-separated, based on which document types contain relevant information for the question."
        )
    )
    user = HumanMessage(content=question)
    resp = llm.invoke([system, user])
    text = resp.content.lower()
    allowed = {"compliance", "risk", "internal"}
    parts = [x.strip() for x in text.split(",") if x.strip() in allowed]
    if not parts:
        parts = ["compliance", "risk", "internal"]
    return {"doc_types": parts, "execution_path": path}

def expand_query(state: State):
    question = state["question"]
    path = state.get("execution_path", [])
    path.append("query_expander")
    ner_system = SystemMessage(
        content=(
            "You are a Named Entity Recognition (NER) expert for compliance and financial documents.\n"
            "Extract the most important words, phrases, and entities from the user's question.\n"
            "Focus on:\n"
            "- Key terms (e.g., 'market manipulation', 'insider trading')\n"
            "- Regulatory concepts (e.g., 'compliance', 'risk management')\n"
            "- Important nouns and noun phrases\n"
            "- Technical terms specific to finance/compliance\n"
            "\n"
            "Return ONLY a comma-separated list of important words/phrases.\n"
            "Do not include explanations, just the words separated by commas.\n"
            "Example: 'market manipulation, trading abuse, price fixing, regulatory compliance'"
        )
    )
    ner_user = HumanMessage(content=f"Question: {question}\n\nImportant words/phrases:")
    ner_resp = llm.invoke([ner_system, ner_user])
    important_words = ner_resp.content.strip()
    
    if important_words.lower().startswith("important words:"):
        important_words = important_words[16:].strip()
    if important_words.lower().startswith("words:"):
        important_words = important_words[7:].strip()
    if important_words.lower().startswith("entities:"):
        important_words = important_words[10:].strip()
    expand_system = SystemMessage(
        content=(
            "You are a query expansion expert for a compliance document search system.\n"
            "Your task is to expand the user's question using definitions of important terms.\n"
            "\n"
            "For each important word/phrase provided, include:\n"
            "- Its definition in the compliance/regulatory context\n"
            "- Related terms and synonyms\n"
            "- Examples or variations of the concept\n"
            "\n"
            "Then create an expanded query that:\n"
            "- Incorporates the definitions and related terms\n"
            "- Maintains the original question's intent\n"
            "- Is comprehensive but concise (2-3 sentences max)\n"
            "\n"
            "Return ONLY the expanded query text. Do not add explanations or prefixes."
        )
    )
    expand_user = HumanMessage(
        content=(
            f"Original question: {question}\n\n"
            f"Important words/phrases to expand: {important_words}\n\n"
            f"Expanded query:"
        )
    )
    expand_resp = llm.invoke([expand_system, expand_user])
    expanded = expand_resp.content.strip()
    
    if expanded.lower().startswith("expanded query:"):
        expanded = expanded[15:].strip()
    if expanded.lower().startswith("expanded:"):
        expanded = expanded[9:].strip()
    
    return {"expanded_query": expanded, "execution_path": path}

def retrieve_chunks(state: State):
    query = state.get("expanded_query") or state["question"]
    doc_types = state["doc_types"]
    path = state.get("execution_path", [])
    path.append("retriever")
    all_chunks = []

    for doc_type in doc_types:
        url = SERVERS.get(doc_type)
        if not url:
            continue
        try:
            response = requests.get(url, params={"query": query}, timeout=REQUEST_TIMEOUT)
            if response.status_code != 200:
                print(f"Warning: {doc_type} server returned status {response.status_code}")
                continue
            data = response.json()
            chunks = data.get("chunks", [])
            for ch in chunks:
                meta = ch.get("metadata", {}) or {}
                meta["server_type"] = doc_type
                meta.setdefault("doc_name", meta.get("doc_name", "unknown_document"))
                meta.setdefault("page", meta.get("page", None))
                meta.setdefault("chunk_id", meta.get("chunk_id", None))
                all_chunks.append(
                    {
                        "content": ch.get("content", ""),
                        "metadata": meta,
                    }
                )
        except Exception as e:
            continue

    top_chunks = all_chunks[:5]
    return {"raw_context": top_chunks, "execution_path": path}

def validate_context(state: State):
    question = state["question"]
    context = state["raw_context"]
    path = state.get("execution_path", [])
    path.append("validator")

    if not context:
        return {
            "validated_context": [],
            "audit_trail": "No contextual evidence found.",
            "execution_path": path
        }

    joined = "\n\n".join(
        f"[{i+1}] {c['metadata']['doc_name']} (server={c['metadata']['server_type']})\n{c['content']}"
        for i, c in enumerate(context)
    )

    system = SystemMessage(
        content=(
            "You are a compliance auditor. Review the context for:\n"
            "- ambiguity\n- contradictions\n- incomplete policy\n"
            "Write a short audit statement (<250 words)."
        )
    )
    user = HumanMessage(content=f"User question:\n{question}\n\nContext:\n{joined}")
    resp = llm.invoke([system, user])

    return {
        "validated_context": context,
        "audit_trail": resp.content.strip(),
        "execution_path": path
    }

def generate_answer(state: State):
    question = state["question"]
    context = state["validated_context"]
    path = state.get("execution_path", [])
    path.append("generator")

    if not context:
        return {
            "answer": (
                "There is no internal documentation available for this topic. "
                "Please consult the official compliance office or policy owner."
            ),
            "execution_path": path
        }

    context_text = "\n\n".join(
        f"[{i+1}] {c['metadata']['doc_name']} (server={c['metadata']['server_type']})\n{c['content']}"
        for i, c in enumerate(context)
    )

    system = SystemMessage(
        content=(
            "You are a compliance assistant.\n"
            "- Answer strictly using the provided content.\n"
            "- If unclear, say so and advise escalation.\n"
        )
    )
    user = HumanMessage(
        content=f"User question:\n{question}\n\nContext:\n{context_text}"
    )
    resp = llm.invoke([system, user])

    return {"answer": resp.content.strip(), "execution_path": path}

def generate_explanation(state: State):
    question = state["question"]
    answer = state["answer"]
    path = state.get("execution_path", [])
    path.append("explainer")

    system = SystemMessage(
        content="Rewrite the answer in plain language for non-experts."
    )
    user = HumanMessage(
        content=f"Q: {question}\n\nA: {answer}"
    )
    resp = llm.invoke([system, user])

    return {"explanation": resp.content.strip(), "execution_path": path}

def trace_sources(state: State):
    context = state["validated_context"]
    path = state.get("execution_path", [])
    path.append("citation_tracker")
    
    sources = [
        {
            "server_type": c["metadata"]["server_type"],
            "doc_name": c["metadata"]["doc_name"],
            "page": c["metadata"]["page"],
            "chunk_id": c["metadata"]["chunk_id"],
            "text": c["content"],
        }
        for c in context
    ]
    return {"sources": sources, "execution_path": path}

def score_confidence(state: State):
    question = state["question"]
    answer = state["answer"]
    audit = state["audit_trail"]
    context = state["validated_context"]
    path = state.get("execution_path", [])
    path.append("confidence_scorer")
    snippets = len(context)

    system = SystemMessage(
        content=(
            "Return ONLY a confidence score in [0,1]. "
            "Use:\n- context completeness\n- clarity\n- ambiguity\n"
            "Base your score on the top 5 retrieved chunks only.\n"
        )
    )
    user = HumanMessage(
        content=(
            f"Q: {question}\nA: {answer}\nAudit: {audit}\n"
            f"Number of chunks used: {snippets} (top 5)\n"
        )
    )
    raw = llm.invoke([system, user]).content.strip()
    try:
        score = float(raw)
    except:
        score = 0.5
    score = max(0, min(1, score))
    return {"confidence": score, "execution_path": path}

graph = StateGraph(State)

graph.add_node("analyzer", analyze_query)
graph.add_node("query_expander", expand_query)
graph.add_node("retriever", retrieve_chunks)
graph.add_node("validator", validate_context)
graph.add_node("generator", generate_answer)
graph.add_node("explainer", generate_explanation)
graph.add_node("citation_tracker", trace_sources)
graph.add_node("confidence_scorer", score_confidence)

graph.set_entry_point("analyzer")
graph.add_edge("analyzer", "query_expander")
graph.add_edge("query_expander", "retriever")
graph.add_edge("retriever", "validator")
graph.add_edge("validator", "generator")
graph.add_edge("generator", "explainer")
graph.add_edge("explainer", "citation_tracker")
graph.add_edge("citation_tracker", "confidence_scorer")
graph.add_edge("confidence_scorer", END)

app = graph.compile()

def export_graph_png(output_path: str = "graph_structure.png"):
    try:
        graph_structure = app.get_graph()
        
        try:
            import graphviz
            from IPython.display import Image, display
            
            dot = graphviz.Digraph(comment='Compliance Agent Graph')
            dot.attr(rankdir='LR', size='12,8')
            dot.attr('node', shape='box', style='rounded', fontname='Arial')
            
            nodes = graph_structure.nodes
            for node_id in nodes:
                label = node_id.replace('_', ' ').title()
                dot.node(node_id, label)
            
            edges = graph_structure.edges
            for edge in edges:
                source = edge.source if hasattr(edge, 'source') else str(edge).split('->')[0].strip()
                target = edge.target if hasattr(edge, 'target') else str(edge).split('->')[1].strip()
                if target == 'END':
                    target = '__end__'
                    dot.node('__end__', 'END', shape='ellipse')
                dot.edge(source, target)
            
            if hasattr(graph_structure, 'first') and graph_structure.first:
                entry = graph_structure.first
                dot.node('__start__', 'START', shape='ellipse', style='filled', fillcolor='lightgreen')
                dot.edge('__start__', entry)
            
            dot.render(output_path, format='png', cleanup=True)
            print(f"Graph exported to {output_path}")
            return output_path
            
        except ImportError:
            try:
                import matplotlib.pyplot as plt
                import matplotlib.patches as mpatches
                from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
                
                fig, ax = plt.subplots(figsize=(14, 10))
                ax.set_xlim(0, 10)
                ax.set_ylim(0, 10)
                ax.axis('off')
                
                nodes = list(graph_structure.nodes.keys())
                node_positions = {}
                x_positions = [1, 2.5, 4, 5.5, 7, 8.5, 10]
                
                for i, node_id in enumerate(nodes):
                    if i < len(x_positions):
                        node_positions[node_id] = (x_positions[i], 5)
                
                for node_id, (x, y) in node_positions.items():
                    label = node_id.replace('_', ' ').title()
                    box = FancyBboxPatch((x-0.4, y-0.3), 0.8, 0.6,
                                        boxstyle="round,pad=0.1",
                                        facecolor='lightblue',
                                        edgecolor='black',
                                        linewidth=2)
                    ax.add_patch(box)
                    ax.text(x, y, label, ha='center', va='center', fontsize=10, fontweight='bold')
                
                edges = graph_structure.edges
                for edge in edges:
                    source = edge.source if hasattr(edge, 'source') else str(edge).split('->')[0].strip()
                    target = edge.target if hasattr(edge, 'target') else str(edge).split('->')[1].strip()
                    
                    if source in node_positions and target in node_positions:
                        x1, y1 = node_positions[source]
                        x2, y2 = node_positions[target]
                        arrow = FancyArrowPatch((x1+0.4, y1), (x2-0.4, y2),
                                               arrowstyle='->', lw=2, color='black')
                        ax.add_patch(arrow)
                
                ax.text(5, 9, 'Compliance Agent Graph Structure', 
                       ha='center', va='center', fontsize=16, fontweight='bold')
                
                plt.tight_layout()
                plt.savefig(output_path, dpi=300, bbox_inches='tight')
                plt.close()
                print(f"Graph exported to {output_path}")
                return output_path
                
            except ImportError:
                print("Error: Neither graphviz nor matplotlib is available.")
                print("Please install one of them:")
                print("  pip install graphviz")
                print("  OR")
                print("  pip install matplotlib")
                return None
                
    except Exception as e:
        print(f"Error exporting graph: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "export_graph":
    export_graph_png()
    sys.exit(0)

def run_query(q: str) -> State:
    return app.invoke({"question": q, "execution_path": []})

def create_artifact_file() -> Path:
    artifacts_dir = Path("artifacts")
    artifacts_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    artifact_file = artifacts_dir / f"trace_{timestamp}.json"
    
    session_data = {
        "session_start": datetime.now().isoformat(),
        "queries": []
    }
    
    with open(artifact_file, 'w') as f:
        json.dump(session_data, f, indent=2)
    
    return artifact_file

def log_trace(artifact_file: Path, question: str, result: State):
    with open(artifact_file, 'r') as f:
        data = json.load(f)
    
    trace_entry = {
        "timestamp": datetime.now().isoformat(),
        "question": question,
        "expanded_query": result.get("expanded_query", ""),
        "doc_types": result.get("doc_types", []),
        "answer": result.get("answer", ""),
        "explanation": result.get("explanation", ""),
        "sources": result.get("sources", []),
        "audit_trail": result.get("audit_trail", ""),
        "confidence": result.get("confidence", 0.0),
        "num_chunks_retrieved": len(result.get("raw_context", [])),
        "num_chunks_used": len(result.get("validated_context", []))
    }
    
    data["queries"].append(trace_entry)
    data["session_end"] = datetime.now().isoformat()
    
    with open(artifact_file, 'w') as f:
        json.dump(data, f, indent=2)

if __name__ == "__main__":
    print("=" * 60)
    print("Compliance Query Agent")
    print("=" * 60)
    print("Type 'quit' or 'exit' to stop the agent.\n")
    
    artifact_file = create_artifact_file()
    print(f"Artifact file created: {artifact_file}\n")
    
    while True:
        query = input("Enter your query: ").strip()
        
        if query.lower() in ['quit', 'exit', 'q']:
            print(f"\nGoodbye! Artifact file saved: {artifact_file}")
            break
        
        if not query:
            print("Error: Query cannot be empty. Please try again.\n")
            continue
        
        print(f"\nProcessing query: {query}")
        print("This may take a moment...\n")
        
        try:
            result = run_query(query)
            log_trace(artifact_file, query, result)

            print("\n" + "=" * 60)
            print("ANSWER")
            print("=" * 60)
            print(result["answer"])

            print("\n" + "=" * 60)
            print("EXPLANATION")
            print("=" * 60)
            print(result["explanation"])

            print("\n" + "=" * 60)
            print("SOURCES")
            print("=" * 60)
            print(json.dumps(result["sources"], indent=2))

            print("\n" + "=" * 60)
            print("AUDIT TRAIL")
            print("=" * 60)
            print(result["audit_trail"])

            print("\n" + "=" * 60)
            print("CONFIDENCE SCORE")
            print("=" * 60)
            print(result["confidence"])
            print("\n" + "=" * 60 + "\n")
        except KeyboardInterrupt:
            print("\n\nInterrupted by user. Goodbye!")
            break
        except Exception as e:
            print(f"\nError processing query: {e}")
            print("Please try again.\n")
            continue

