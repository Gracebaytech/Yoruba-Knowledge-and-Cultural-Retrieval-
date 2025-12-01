import streamlit as st
from typing import Dict, List
from utils.rag_system import YorubaRAG  # Wrapper for AdvancedRAG


# =============================================================================
# 1. STREAMLIT PAGE CONFIG + CSS
# =============================================================================

def setup_page_config():
    st.set_page_config(
        page_title="Yorùbá Linguistic Knowledge System",
        page_icon="📚",
        layout="wide",
        initial_sidebar_state="expanded"
    )


def setup_custom_css():
    st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem; color: #1f77b4; text-align: center; 
        margin-bottom: 2rem; font-weight: bold;
    }
    .yoruba-text {
        font-size: 1.2rem; line-height: 1.6; color: #2e4057;
    }
    .answer-box {
        background-color: #f0f8ff; padding: 1.5rem; border-radius: 10px;
        border-left: 5px solid #1f77b4; margin: 1rem 0;
    }
    .source-box {
        background-color: #fffaf0; padding: 1rem; border-radius: 8px;
        border-left: 3px solid #ff7f0e; margin: .5rem 0;
    }
    .domain-tag {
        background-color: #1f77b4; color: white; padding: .3rem .8rem;
        border-radius: 15px; font-size: .8rem; display: inline-block; 
        margin: .2rem;
    }
    .metric-box {
        background-color: #f8f9fa; padding: 1rem; border-radius: 8px;
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)


# =============================================================================
# 2. UI SECTIONS
# =============================================================================

def render_header():
    st.markdown(
        "<div class='main-header'>📚 Ọ̀RỌ̀ ỌMỌ YORÙBÁ - Yorùbá Linguistic Knowledge System</div>",
        unsafe_allow_html=True
    )
    st.markdown(
        "<div style='text-align:center;color:#666;margin-bottom:2rem;'>"
        "Iṣẹ́ ìwádìí èdè àti àṣà Yorùbá • Yorùbá Language and Cultural Research System"
        "</div>", unsafe_allow_html=True
    )


def render_sidebar():
    with st.sidebar:
        st.header("📖 Nípa Ẹ̀rọ Yìí")
        st.markdown("""
        **Ọ̀RỌ̀ ỌMỌ YORÙBÁ** jẹ́ ẹ̀rọ ìwádìí tó ń lo ìmọ̀ ẹ̀rọ láti:
        - 🎯 Dáhùn ìbéèrè nípa èdè àti àṣà Yorùbá  
        - 📚 Fúnni ní àwọn àkọsílẹ̀ àti ìwé gẹ́gẹ́ bí ìtọ́kasí  
        - 🔤 Ṣàtúnṣe àwọn àmì ìyọ̀tọ́rọ̀ (diacritics)  
        - 🌍 Ṣe àtìlẹ́yìn fún ìpamọ́ ìmọ̀ èdè Yorùbá  
        """)
        st.markdown("---")

        st.header("🎯 Àwọn Ẹ̀ka Ìmọ̀")
        domains = ["Ìṣẹ̀ṣe • Religion", "Àṣà • Culture", "Ìṣèlú • Politics",
                   "Ìṣeré • Entertainment", "Ìṣòwò • Social Life"]
        for d in domains:
            st.markdown(f"• {d}")

        st.markdown("---")
        st.caption("🎓 Ẹ̀kọ́ Ọ̀jọ̀gbọ́n: *Ìwádìí Ìmọ̀ Ẹ̀rọ fún Ìpamọ́ Èdè Yorùbá*" )


def render_query_input():
    st.markdown("### 🎯 Ṣe Ìbéèrè Rẹ")
    query = st.text_area(
        "Tẹ ìbéèrè rẹ ní Yorùbá tàbí Gẹ̀ẹ́sì:",
        height=100,
        key="query_input",
        placeholder="Àpẹẹrẹ: 'Kí ni ìtumọ̀ ẹbọ nínú ìṣẹ̀ṣe Yorùbá?'"
    )

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        submitted = st.button("🚀 Dáhùn Ìbéèrè • Generate Answer", use_container_width=True)

    return query, submitted


def render_answer_section(response: Dict):
    st.markdown("### 💡 Ìdáhùn • Answer")
    with st.container():
        st.markdown('<div class="answer-box">', unsafe_allow_html=True)
        st.markdown(f'<div class="yoruba-text">{response.get("answer","")}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("⏱️ Ìgbà Ìdáhùn", f"{response.get('response_time', 0):.2f}s")
    with col2:
        st.metric("📚 Ìwé Tí a Rí", response.get("retrieved_count", 0))
    with col3:
        st.metric("🎯 Ẹ̀ka Ìmọ̀", response.get("domain", "N/A"))
    with col4:
        st.metric("📊 Ìwọn Ìdáhùn", f"{len(response.get('answer', '').split())} ọ̀rọ̀")


def render_source_documents(sources: List[Dict]):
    st.markdown("### 📚 Àwọn Orísun Ìwé • Source Documents")

    if not sources:
        st.info("🔍 Ko sí ìwé tí a rí fún ìbéèrè yìí.")
        return

    for i, src in enumerate(sources, 1):
        with st.expander(f"Orísun {i}: {src.get('domain','N/A')} - {src.get('source','Unknown')}",
                         expanded=(i == 1)):
            col1, col2 = st.columns([3, 1])

            with col1:
                content_preview = src.get("content", "")[:300]
                st.markdown(f"**Àkọsọ:** {content_preview}...")

            with col2:
                st.markdown(
                    f"**Ẹ̀ka:** <span class='domain-tag'>{src.get('domain','N/A')}</span>",
                    unsafe_allow_html=True
                )
                st.markdown(f"**Orísun:** {src.get('source','Unknown')}")
                if src.get("url"):
                    st.markdown(f"**URL:** [Ṣe àwárí]({src['url']})")

            st.progress(min(src.get("score", 0.5), 1.0),
                        text=f"Ìjọra: {src.get('score',0):.2f}")


def render_example_queries():
    st.markdown("### 💡 Àwọn Ìbéèrè Àpẹẹrẹ")
    examples = {
        "Ìṣẹ̀ṣe": [
            "Kí ni ìtumọ̀ ẹbọ nínú ìṣẹ̀ṣe Yorùbá?",
            "Àwọn oríṣà mẹ́ta pàtàkì jùlọ wo ni?",
            "Kí ló jẹ́ ka mọ̀ nípa àṣẹ nínú Yorùbá?"
        ],
        "Àṣà": [
            "Kí ni ìtumọ̀ ọmọlúàbí?",
            "Ṣàlàyé ìlànà ìgbéyàwó Yorùbá.",
            "Darukọ àwọn òwe mẹ́ta tó wọ́pọ̀ jùlọ."
        ]
    }

    for cat, qs in examples.items():
        with st.expander(cat):
            for q in qs:
                if st.button(q, key=f"example_{hash(q)}", use_container_width=True):
                    st.session_state.query_input = q
                    st.rerun()


def render_footer():
    st.markdown("---")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**📞 Ọ̀rọ̀ Ìbánisọ̀rọ̀**\ninfo@yoruba.ai\n+234-XXX-XXXX")
    with col2:
        st.markdown("**🔧 Imọ̀ Ẹ̀rọ**\nGemini 2.5 • Modular RAG\nWeaviate • AfriBERTa")
    with col3:
        st.markdown("**🎓 Ìwádìí**\nÌmọ̀ Ẹ̀rọ fún Èdè Yorùbá\nỌ̀jọ̀gbọ́n [Orúkọ Rẹ]")

    st.caption("© 2024 Ọ̀RỌ̀ ỌMỌ YORÙBÁ • Academic Research System")


# =============================================================================
# 3. MAIN APP
# =============================================================================

def main():
    setup_page_config()
    setup_custom_css()
    render_header()
    render_sidebar()

    # Initialize RAG system once
    if "rag_system" not in st.session_state:
        with st.spinner("🚀 Ẹ̀rọ ń gbé kalẹ̀... Initializing System..."):
            st.session_state.rag_system = YorubaRAG()
    if "last_response" not in st.session_state:
        st.session_state.last_response = None

    col1, col2 = st.columns([2, 1])

    # Main Query + Results
    with col1:
        query, submitted = render_query_input()

        if submitted and query.strip():
            with st.spinner("🔍 Ẹ̀rọ ń wádìí àwọn ìwé... Searching documents..."):
                resp = st.session_state.rag_system.query(query)
                st.session_state.last_response = resp

        # Show previous or new response
        if st.session_state.last_response:
            render_answer_section(st.session_state.last_response)
            render_source_documents(st.session_state.last_response.get("source_documents", []))

    # Right column tools
    with col2:
        render_example_queries()

    render_footer()


if __name__ == "__main__":
    main()
