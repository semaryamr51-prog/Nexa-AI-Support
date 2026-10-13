"""Nexa · redesigned premium Streamlit support workspace.

Single-file UI; safe offline mode; no external logo, image, or API key required.
Run: streamlit run app.py
"""
from __future__ import annotations

import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent / '.env')

from src.agents import route_and_run
from src.classify import classifier_backend
from src.llm import get_provider_status

st.set_page_config(page_title="Nexa | AI Support", page_icon="💠", layout="wide", initial_sidebar_state="expanded")

st.markdown('''
<style>
/* Nexa — custom visual system */
:root { --n-bg:#08111f; --n-card:#101e30; --n-border:#21354a; --n-text:#ecf5ff; --n-muted:#9eafc3; --n-accent:#56e9cd; }
html,body,[data-testid="stApp"],[data-testid="stAppViewContainer"],.stApp {background:#08111f !important;color:#ecf5ff;}
[data-testid="stAppViewContainer"] > .main {background:radial-gradient(ellipse 80% 55% at 75% -7%,rgba(50,126,152,.15),transparent 70%),#08111f !important;}
[data-testid="stHeader"] {background:transparent !important;}
[data-testid="stSidebar"] {background:#0b1828 !important;border-right:1px solid #203248;}
[data-testid="stSidebar"] > div:first-child {padding-top:1.8rem;}
.block-container {max-width:1090px !important;padding-top:2.7rem !important;padding-bottom:6rem !important;}
#MainMenu,footer{visibility:hidden;}
h1,h2,h3,p,div,span {font-family:Inter,ui-sans-serif,system-ui,-apple-system,'Segoe UI',sans-serif;}
.nexa-brand{display:flex;align-items:center;gap:13px;margin:0 0 26px 0;}
.nexa-monogram{height:42px;width:42px;border-radius:14px;background:linear-gradient(135deg,#67f5d9,#4c81f2);color:#06131e;display:flex;align-items:center;justify-content:center;font-size:26px;font-weight:800;box-shadow:0 4px 20px rgba(58,210,191,.12);}
.nexa-wordmark{font-weight:840;font-size:22px;letter-spacing:.15em;color:#f2f9ff;line-height:1.15;}
.nexa-brand-sub{font-weight:550;letter-spacing:.17em;color:#869fb5;font-size:10px;margin-top:4px;}
.hero{padding:38px 43px 34px;border:1px solid #294055;border-radius:22px;background:radial-gradient(ellipse 53% 130% at 98% 20%,rgba(38,175,182,.2),transparent 70%),linear-gradient(135deg,#17283b,#0c1a2b 70%);margin:0 0 23px;position:relative;overflow:hidden;}
.hero:after{content:'✦';position:absolute;right:44px;top:17px;color:rgba(96,232,201,.18);font-size:155px;font-weight:400;line-height:1;pointer-events:none;}
.eyebrow{font-size:11px;letter-spacing:.23em;color:#67ead3;font-weight:800;text-transform:uppercase;}
.hero h1{font-size:clamp(30px,4vw,47px);font-weight:770;color:#f7fbff;letter-spacing:-.05em;line-height:1.17;margin:15px 0;max-width:670px;}
.hero h1 span{color:#6fe5d5;}
.hero p{color:#b4c4d8;font-size:14px;max-width:590px;line-height:1.85;margin:0;}
.section-title{color:#f2f7ff;font-size:15px;font-weight:760;letter-spacing:-.015em;margin:24px 0 11px;}
.small-label{color:#7e97af;text-transform:uppercase;letter-spacing:.16em;font-size:10px;font-weight:800;}
.features{display:flex;gap:9px;flex-wrap:wrap;margin-top:22px;}
.feature{font-size:11px;font-weight:700;letter-spacing:.02em;background:rgba(93,228,209,.07);border:1px solid rgba(106,230,212,.21);color:#91e5d8;padding:7px 11px;border-radius:999px;}
.quick-card{min-height:118px;border-radius:15px;padding:18px 18px;background:linear-gradient(140deg,#101f31,#0d1b2b);border:1px solid #24374d;}
.quick-index{color:#65dfd3;font-size:12px;font-weight:800;margin-bottom:7px;}
.quick-head{color:#e8f3ff;font-size:14px;font-weight:750;margin:0 0 5px;}
.quick-sub{font-size:12px;color:#90a6bd;line-height:1.5;margin:0;}
.sidebar-panel{border:1px solid #263c50;border-radius:14px;padding:16px 14px;margin:16px 0;background:#101f30;}
.sidebar-heading{color:#8ea4b8;font-size:10px;text-transform:uppercase;letter-spacing:.16em;font-weight:850;margin-bottom:13px;}
.sidebar-value{color:#eaf6ff;font-weight:700;font-size:13px;}
.green-dot{display:inline-block;height:7px;width:7px;background:#5ee3ba;border-radius:50%;box-shadow:0 0 10px #5ee3ba6e;margin-right:8px;vertical-align:middle;}
.muted-text{font-size:11px;color:#829bb0;line-height:1.65;margin-top:9px;}
.live-line{border-top:1px solid #253b50;margin:11px 0;padding-top:10px;display:flex;align-items:center;justify-content:space-between;gap:10px;color:#9fb1c5;font-size:12px;}
.live-line strong{color:#def7f4;font-weight:730;}
.demo-note{border:1px solid #34455c;border-radius:12px;padding:12px 13px;color:#a5b8c8;background:rgba(83,123,152,.08);font-size:11px;line-height:1.7;margin-top:20px;}
[data-testid="stChatMessage"]{background:#101f30;border:1px solid #283b50;border-radius:17px;margin:12px 0;padding:9px 14px;}
[data-testid="stChatMessage"] p{color:#e7f1f8;line-height:1.75;}
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] a {color:#73e8d7;}
[data-testid="stChatInput"] {border-radius:16px !important;border:1px solid #38546b !important;background:#112136 !important;}
[data-testid="stChatInput"] textarea{color:#f5f8fb !important;}
.stButton>button{border:1px solid #2c4b61 !important;background:#14273a !important;color:#d9eff4 !important;border-radius:12px !important;min-height:42px !important;font-weight:650 !important;transition:all .15s ease;}
.stButton>button:hover{border-color:#62e7d0 !important;color:#73ead6 !important;background:#183347 !important;}
[data-testid="stSidebar"] .stButton>button{width:100%;}
[data-testid="stDownloadButton"]>button{border:1px solid #2c4b61 !important;background:#14273a !important;color:#e0f2fa !important;border-radius:12px !important;width:100%;}
[data-testid="stMetricValue"]{color:#6be4cf !important;}
[data-testid="stMetricLabel"]{color:#95aac0 !important;}
[data-testid="stProgressBar"]>div>div{background:#5be7cb !important;}
[data-testid="stExpander"]{border:1px solid #273d50;border-radius:12px;background:#0c1a2a;}
hr{border-color:#263a4c !important;}
@media(max-width:760px){.block-container{padding-top:1.5rem !important;padding-left:1rem !important;padding-right:1rem !important;}.hero{padding:26px 22px;}.hero:after{display:none;}.hero h1{font-size:33px;}}
</style>
''', unsafe_allow_html=True)

if 'history' not in st.session_state:
    st.session_state.history = []
if 'requests' not in st.session_state:
    st.session_state.requests = 0


def export_chat() -> str:
    rows = ['# Nexa Support · Chat Export', '', '> Demo system: billing and calendar records are simulated.', '']
    for turn in st.session_state.history:
        rows += [f"## {'You' if turn['role'] == 'user' else 'Nexa'}", '', turn['content'], '']
    return '\n'.join(rows)


last = next((entry['trace'] for entry in reversed(st.session_state.history)
             if entry['role'] == 'assistant' and entry.get('trace')), None)

with st.sidebar:
    st.markdown('''<div class="nexa-brand"><div class="nexa-monogram">✦</div><div><div class="nexa-wordmark">NEXA</div><div class="nexa-brand-sub">AI SUPPORT STUDIO</div></div></div>''', unsafe_allow_html=True)
    if st.button('＋  New conversation', use_container_width=True, key='new_chat'):
        st.session_state.history = []
        st.session_state.requests = 0
        st.rerun()
    st.markdown('''<div class="sidebar-panel"><div class="sidebar-heading">SYSTEM STATUS</div><div class="sidebar-value"><span class="green-dot"></span> Workspace online</div><div class="muted-text">Routing, retrieval, and specialist workflows remain available without an API key.</div></div>''', unsafe_allow_html=True)
    st.markdown('**Workspace activity**')
    col1, col2 = st.columns(2)
    col1.metric('Messages', st.session_state.requests)
    col2.metric('Agents', '04')
    st.divider()
    st.markdown('<div class="sidebar-heading">CONFIGURATION</div>', unsafe_allow_html=True)
    st.caption('**Reply engine**')
    st.write(get_provider_status())
    st.caption('**Intent classifier**')
    st.write(classifier_backend())
    st.divider()
    st.markdown('<div class="sidebar-heading">LAST ROUTING DECISION</div>', unsafe_allow_html=True)
    if last:
        st.markdown(f"**{last['agent'].title()} agent**")
        st.progress(max(0.0, min(1.0, last['confidence'])), text=f"Confidence {last['confidence']:.0%}")
        st.caption(f"Predicted: {last['intent']} · Responded via: {last['provider']}")
        st.caption('Tools: ' + ', '.join(last['tool_calls']))
    else:
        st.caption('Send a message to inspect the routing decisions.')
    st.divider()
    st.download_button('↓  Export conversation', data=export_chat(), file_name='nexa_conversation.md',
                       mime='text/markdown', use_container_width=True, disabled=not bool(st.session_state.history))
    st.markdown('''<div class="demo-note"><strong>Demo environment</strong><br>Invoices and calendar slots are simulated. No real purchases, refunds, or reservations take place here.</div>''', unsafe_allow_html=True)

st.markdown('''<section class="hero"><div class="eyebrow">✦ &nbsp; INTELLIGENT SUPPORT WORKSPACE</div><h1>Every question.<br><span>The right expert.</span></h1><p>A context-aware assistant that routes your requests to specialist agents, searches trusted knowledge, and explains how it reached an answer.</p><div class="features"><span class="feature">◉  4 specialized agents</span><span class="feature">⌁  Grounded knowledge search</span><span class="feature">↗  Offline ready</span></div></section>''', unsafe_allow_html=True)

prompt = None
if not st.session_state.history:
    st.markdown('<div class="section-title">Choose a starting point</div>', unsafe_allow_html=True)
    suggestions = [
        ('01', 'Payments & refunds', 'Understand policies and invoices.', 'What is the refund policy?'),
        ('02', 'Technical help', 'Solve account and playback issues.', 'My course video is not loading. What should I do?'),
        ('03', 'Sessions & booking', 'Explore office hours and mentoring.', 'Can I book office hours next Monday?'),
        ('04', 'Course information', 'Find learning and certificate details.', 'Do I receive a certificate when I finish a course?'),
    ]
    for i in (0, 2):
        cols = st.columns(2, gap='medium')
        for j, column in enumerate(cols):
            idx, heading, detail, sample = suggestions[i + j]
            with column:
                st.markdown(f'<div class="quick-card"><div class="quick-index">{idx} / SUPPORT</div><div class="quick-head">{heading}</div><p class="quick-sub">{detail}</p></div>', unsafe_allow_html=True)
                if st.button(f'Ask about {heading.lower()}  ↗', key=f'prompt_{idx}', use_container_width=True):
                    prompt = sample
    st.markdown('<div class="section-title">Or ask anything below</div>', unsafe_allow_html=True)
else:
    st.markdown('<div class="section-title">Conversation</div>', unsafe_allow_html=True)

for entry in st.session_state.history:
    with st.chat_message(entry['role'], avatar='🤖' if entry['role'] == 'assistant' else '👤'):
        st.markdown(entry['content'])
        if entry['role'] == 'assistant' and entry.get('trace'):
            trace = entry['trace']
            with st.expander(f"Agent activity  ·  {trace['agent'].title()}"):
                a, b, c = st.columns(3)
                a.metric('Intent', trace['intent'].title())
                b.metric('Confidence', f"{trace['confidence']:.0%}")
                c.metric('Provider', trace['provider'])
                st.caption('Tools: ' + ', '.join(trace['tool_calls']))
                if trace.get('sources'):
                    st.caption('Knowledge sources: ' + ' · '.join(trace['sources']))

written = st.chat_input('Ask your question…', max_chars=4000)
message = prompt or written
if message and message.strip():
    st.session_state.history.append({'role': 'user', 'content': message})
    with st.chat_message('user', avatar='👤'):
        st.markdown(message)
    with st.chat_message('assistant', avatar='🤖'):
        with st.spinner('Finding the best specialist and trusted answers…'):
            try:
                result = route_and_run(message, history=st.session_state.history[:-1])
                answer = result['reply']
                trace = {k: result[k] for k in ('intent', 'confidence', 'agent', 'tool_calls', 'sources', 'provider')}
            except Exception:
                answer = 'Something interrupted that request. You can retry, or use offline mode without configuring an LLM provider.'
                trace = None
        st.markdown(answer)
        if trace:
            with st.expander(f"Agent activity  ·  {trace['agent'].title()}"):
                st.write(f"**Intent:** {trace['intent']} · **Confidence:** {trace['confidence']:.0%} · **Provider:** {trace['provider']}")
                st.caption('Tools: ' + ', '.join(trace['tool_calls']))
                if trace['sources']:
                    st.caption('Knowledge sources: ' + ' · '.join(trace['sources']))
    st.session_state.history.append({'role': 'assistant', 'content': answer, 'trace': trace})
    st.session_state.requests += 1
    st.rerun()
