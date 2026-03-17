"""
Token-Ring Mutual Exclusion with Replication - Complete Streamlit App
=====================================================================
Demonstrates:
1. Logical ring of MSSs where MHs make requests for tokens
2. Request broadcasting to all MSSs and priority-based granting
3. Replicated request logs at each MSS with priorities
4. Queue state after requests have been served
"""

import streamlit as st
import pandas as pd
import numpy as np
import math
import random
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

# ═══════════════════════════════════════════════════════════════
#                      MODEL CLASSES
# ═══════════════════════════════════════════════════════════════

class LamportClock:
    """Lamport logical clock for distributed event ordering"""
    def __init__(self):
        self.time = 0

    def increment(self):
        self.time += 1
        return self.time

    def update(self, received_time):
        self.time = max(self.time, received_time) + 1
        return self.time


@dataclass
class Request:
    """Request for critical section access"""
    mh_id: str
    source_mss_id: int
    priority: int
    timestamp: int
    status: str = "PENDING"
    request_id: str = field(default="", init=False)

    def __post_init__(self):
        self.request_id = f"REQ_{self.mh_id}_T{self.timestamp}"

    def to_dict(self):
        return {
            "Request ID": self.request_id,
            "Mobile Host": self.mh_id,
            "Source MSS": f"MSS_{self.source_mss_id}",
            "Priority": self.priority,
            "Lamport Time": self.timestamp,
            "Status": self.status,
        }


class MobileHost:
    """Mobile Host that connects to an MSS and requests critical section"""
    def __init__(self, mh_id: str, current_mss, base_priority: int = None):
        self.id = mh_id
        self.current_mss = current_mss
        self.base_priority = base_priority if base_priority else random.randint(1, 10)
        self.in_cs = False
        self.current_request: Optional[Request] = None

    def request_cs(self):
        if self.current_request and self.current_request.status == "PENDING":
            return None
        request = self.current_mss.receive_request_from_mh(self, self.base_priority)
        self.current_request = request
        return request

    def enter_cs(self):
        self.in_cs = True
        if self.current_request:
            self.current_request.status = "GRANTED"

    def exit_cs(self):
        self.in_cs = False
        if self.current_request:
            self.current_request.status = "COMPLETED"
            self.current_request = None


class MSS:
    """Mobile Support Station — forms a logical ring, manages token and requests"""
    def __init__(self, mss_id: int):
        self.id = mss_id
        self.next_mss: Optional["MSS"] = None
        self.prev_mss: Optional["MSS"] = None
        self.has_token = False
        self.clock = LamportClock()

        self.replicated_log: List[Request] = []
        self.local_queue: List[Request] = []
        self.global_queue: List[Request] = []
        self.mobile_hosts: List[MobileHost] = []

        self.messages_sent = 0
        self.messages_received = 0
        self.grants_made = 0

    def add_mh(self, mh: MobileHost):
        self.mobile_hosts.append(mh)
        mh.current_mss = self

    def receive_request_from_mh(self, mh: MobileHost, priority: int) -> Request:
        self.clock.increment()
        req = Request(
            mh_id=mh.id,
            source_mss_id=self.id,
            priority=priority,
            timestamp=self.clock.time,
        )
        self.replicated_log.append(req)
        self.local_queue.append(req)
        self.global_queue.append(req)
        self._sort_queue()
        self.broadcast(req)
        return req

    def broadcast(self, req: Request):
        cur = self.next_mss
        while cur and cur.id != self.id:
            cur.receive_replicated(req)
            self.messages_sent += 1
            cur = cur.next_mss

    def receive_replicated(self, req: Request):
        self.clock.update(req.timestamp)
        self.replicated_log.append(req)
        self.global_queue.append(req)
        self._sort_queue()
        self.messages_received += 1

    def _sort_queue(self):
        self.global_queue.sort(key=lambda r: (-r.priority, r.timestamp))

    def grant_token(self) -> Optional[Request]:
        if not self.has_token:
            return None
        local_pending = [
            r for r in self.global_queue
            if r.source_mss_id == self.id and r.status == "PENDING"
        ]
        if not local_pending:
            return None
        granted = local_pending[0]
        granted.status = "GRANTED"
        if granted in self.local_queue:
            self.local_queue.remove(granted)
        self.grants_made += 1
        return granted

    def pass_token(self):
        if not self.has_token:
            return
        self.has_token = False
        self.next_mss.has_token = True
        self.next_mss.clock.increment()
        self.messages_sent += 1

    def stats(self):
        return {
            "MSS": f"MSS_{self.id}",
            "Sent": self.messages_sent,
            "Received": self.messages_received,
            "Grants": self.grants_made,
            "MHs": len(self.mobile_hosts),
            "Pending": len([r for r in self.local_queue if r.status == "PENDING"]),
        }


# ═══════════════════════════════════════════════════════════════
#                RING TOPOLOGY & TOKEN MANAGER
# ═══════════════════════════════════════════════════════════════

class RingTopology:
    def __init__(self, n: int):
        self.n = n
        self.nodes: List[MSS] = [MSS(i) for i in range(n)]
        for i in range(n):
            self.nodes[i].next_mss = self.nodes[(i + 1) % n]
            self.nodes[i].prev_mss = self.nodes[(i - 1) % n]
        self.nodes[0].has_token = True

    def token_holder(self) -> Optional[MSS]:
        for m in self.nodes:
            if m.has_token:
                return m
        return None


class TokenManager:
    def __init__(self, ring: RingTopology):
        self.ring = ring
        self.event_log: List[str] = []
        self.circulations = 0

    def step(self) -> Tuple[Optional[Request], str]:
        holder = self.ring.token_holder()
        if not holder:
            return None, "⚠️ No token holder found"

        granted = holder.grant_token()
        if granted:
            msg = (
                f"✅ MSS_{holder.id} GRANTED token → {granted.mh_id} "
                f"(Priority={granted.priority}, T={granted.timestamp})"
            )
            self.event_log.append(msg)
            return granted, msg

        next_id = holder.next_mss.id
        holder.pass_token()
        self.circulations += 1
        msg = f"➡️ Token passed: MSS_{holder.id} → MSS_{next_id}"
        self.event_log.append(msg)
        return None, msg

    def complete(self, req: Request):
        req.status = "COMPLETED"
        for mss in self.ring.nodes:
            if req in mss.global_queue:
                mss.global_queue.remove(req)
        self.event_log.append(f"🏁 {req.mh_id} completed CS and released token")


# ═══════════════════════════════════════════════════════════════
#                     VISUALIZATION HELPERS
# ═══════════════════════════════════════════════════════════════

import plotly.graph_objects as go


def draw_ring(mss_list: List[MSS]) -> go.Figure:
    """Draw MSS ring with MH labels using Plotly"""
    n = len(mss_list)
    angles = [2 * math.pi * i / n - math.pi / 2 for i in range(n)]
    radius = 2.0

    xs = [radius * math.cos(a) for a in angles]
    ys = [radius * math.sin(a) for a in angles]

    # ── arrows (edges) ──
    arrow_annotations = []
    for i in range(n):
        x0, y0 = xs[i], ys[i]
        x1, y1 = xs[(i + 1) % n], ys[(i + 1) % n]
        # shorten so arrow doesn't overlap node
        dx, dy = x1 - x0, y1 - y0
        length = math.hypot(dx, dy)
        shrink = 0.22
        ax0 = x0 + dx * shrink
        ay0 = y0 + dy * shrink
        ax1 = x1 - dx * shrink
        ay1 = y1 - dy * shrink
        arrow_annotations.append(
            dict(
                ax=ax0, ay=ay0, x=ax1, y=ay1,
                xref="x", yref="y", axref="x", ayref="y",
                showarrow=True,
                arrowhead=3, arrowsize=1.5, arrowwidth=2,
                arrowcolor="#555",
            )
        )

    # ── nodes ──
    colors = ["gold" if m.has_token else "#64b5f6" for m in mss_list]
    sizes = [55 if m.has_token else 42 for m in mss_list]
    symbols = []
    hovers = []
    labels = []
    for m in mss_list:
        tok = " 🔑" if m.has_token else ""
        labels.append(f"MSS_{m.id}{tok}")
        mh_names = ", ".join(f"{mh.id}(P{mh.base_priority})" for mh in m.mobile_hosts) or "—"
        hovers.append(
            f"<b>MSS_{m.id}</b>{tok}<br>"
            f"MHs: {mh_names}<br>"
            f"Local pending: {len(m.local_queue)}"
        )

    node_trace = go.Scatter(
        x=xs, y=ys, mode="markers+text",
        text=labels, textposition="top center",
        textfont=dict(size=13, color="black"),
        hovertext=hovers, hoverinfo="text",
        marker=dict(size=sizes, color=colors,
                    line=dict(width=2, color="black")),
        showlegend=False,
    )

    # ── MH labels around each MSS ──
    mh_annotations = []
    for idx, m in enumerate(mss_list):
        for j, mh in enumerate(m.mobile_hosts):
            offset_angle = angles[idx] + math.pi + (j - len(m.mobile_hosts) / 2) * 0.35
            mx = xs[idx] + 0.55 * math.cos(offset_angle)
            my = ys[idx] + 0.55 * math.sin(offset_angle)
            mh_annotations.append(
                dict(
                    x=mx, y=my, text=f"📱{mh.id}<br>P={mh.base_priority}",
                    showarrow=True, arrowhead=0, arrowwidth=1,
                    arrowcolor="#aaa", ax=xs[idx], ay=ys[idx],
                    font=dict(size=10, color="#333"),
                    bgcolor="#e8f5e9", borderpad=2,
                    xref="x", yref="y", axref="x", ayref="y",
                )
            )

    fig = go.Figure(data=[node_trace])
    fig.update_layout(
        title="<b>Logical Ring of Mobile Support Stations (MSSs)</b>",
        titlefont_size=18,
        annotations=arrow_annotations + mh_annotations,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-3.5, 3.5]),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False,
                   scaleanchor="x", range=[-3.5, 3.5]),
        height=600, plot_bgcolor="white", paper_bgcolor="white",
        margin=dict(l=20, r=20, t=50, b=20),
    )
    return fig


# ═══════════════════════════════════════════════════════════════
#                    SCENARIO BUILDER
# ═══════════════════════════════════════════════════════════════

def build_default_scenario():
    ring = RingTopology(4)
    mhs = []
    config = [
        # (mss_id, mh_id, priority)
        (0, "MH_A", 5),
        (0, "MH_B", 3),
        (1, "MH_C", 8),
        (1, "MH_D", 6),
        (2, "MH_E", 4),
        (2, "MH_F", 7),
        (3, "MH_G", 2),
        (3, "MH_H", 9),
    ]
    for mss_id, mh_id, pri in config:
        mh = MobileHost(mh_id, ring.nodes[mss_id], base_priority=pri)
        ring.nodes[mss_id].add_mh(mh)
        mhs.append(mh)
    return ring, mhs


# ═══════════════════════════════════════════════════════════════
#                       STREAMLIT APP
# ═══════════════════════════════════════════════════════════════

st.set_page_config(page_title="Token-Ring ME Replication", page_icon="🔐", layout="wide")

st.markdown(
    """
    <style>
    .block-container{padding-top:1rem;}
    .header{font-size:2rem;font-weight:700;color:#0d47a1;text-align:center;
            padding:.8rem;background:linear-gradient(90deg,#e3f2fd,#bbdefb);
            border-radius:10px;margin-bottom:1.2rem;}
    .sub{color:#555;text-align:center;margin-bottom:1.5rem;}
    </style>
    <div class="header">🔐 Token-Ring Mutual Exclusion — Replication Scheme</div>
    <p class="sub">MSS-MH Architecture &nbsp;|&nbsp; Request Broadcasting &nbsp;|&nbsp;
    Priority-Based Granting &nbsp;|&nbsp; Replicated Logs &amp; Queues</p>
    """,
    unsafe_allow_html=True,
)

# ── session state init ──
if "ring" not in st.session_state:
    r, m = build_default_scenario()
    st.session_state.ring = r
    st.session_state.mhs = m
    st.session_state.tm = TokenManager(r)
    st.session_state.step = 0
    st.session_state.reqs_made: List[Request] = []

ring: RingTopology = st.session_state.ring
mhs: List[MobileHost] = st.session_state.mhs
tm: TokenManager = st.session_state.tm

# ── sidebar ──
with st.sidebar:
    st.header("⚙️ Controls")

    if st.button("🔄 Reset Everything"):
        r, m = build_default_scenario()
        st.session_state.ring = r
        st.session_state.mhs = m
        st.session_state.tm = TokenManager(r)
        st.session_state.step = 0
        st.session_state.reqs_made = []
        st.rerun()

    st.markdown("---")
    st.subheader("📤 Send Request")
    opts = [f"{mh.id}  (at MSS_{mh.current_mss.id}, P={mh.base_priority})" for mh in mhs]
    sel = st.selectbox("Select Mobile Host", range(len(mhs)), format_func=lambda i: opts[i])
    if st.button("Send Request", use_container_width=True):
        req = mhs[sel].request_cs()
        if req:
            st.session_state.reqs_made.append(req)
            st.success(f"{mhs[sel].id} requested CS")
        else:
            st.warning("Already has a pending request")

    st.markdown("---")
    st.subheader("🔄 Token Circulation")

    c1, c2 = st.columns(2)
    with c1:
        if st.button("▶ Step", use_container_width=True):
            granted, ev = tm.step()
            st.session_state.step += 1
            if granted:
                st.success(f"Granted → {granted.mh_id}")
    with c2:
        if st.button("⏩ ×5", use_container_width=True):
            for _ in range(5):
                tm.step()
                st.session_state.step += 1

    st.markdown("---")
    st.subheader("🏁 Complete CS")
    granted_list = [r for r in st.session_state.reqs_made if r.status == "GRANTED"]
    if granted_list:
        g_opts = [f"{r.mh_id} ({r.request_id})" for r in granted_list]
        g_sel = st.selectbox("Select granted request", range(len(granted_list)),
                             format_func=lambda i: g_opts[i])
        if st.button("Mark Completed", use_container_width=True):
            chosen = granted_list[g_sel]
            tm.complete(chosen)
            # find the MH and exit CS
            for mh in mhs:
                if mh.id == chosen.mh_id:
                    mh.exit_cs()
            st.success(f"{chosen.mh_id} completed CS")
    else:
        st.info("No granted requests to complete")

    st.markdown("---")
    holder = ring.token_holder()
    st.metric("Steps", st.session_state.step)
    st.metric("Token At", f"MSS_{holder.id}" if holder else "—")
    st.metric("Circulations", tm.circulations)

# ═══════════════════════════════════════════════════════════════
#                           TABS
# ═══════════════════════════════════════════════════════════════

tab1, tab2, tab3, tab4 = st.tabs([
    "🔗 1. Ring Topology",
    "📡 2. Request Propagation & Granting",
    "📋 3. Request Logs & Priorities",
    "📊 4. Queues After Service",
])

# ────────── TAB 1: RING TOPOLOGY ──────────
with tab1:
    st.markdown("### 1 — Logical Ring of MSSs with Mobile Hosts")
    st.info(
        "**Architecture:** MSSs (fixed base stations) form a logical ring.  "
        "MHs (mobile devices) attach wirelessly to their local MSS.  "
        "A **single token** circulates clockwise among MSSs.  "
        "The gold node holds the token."
    )

    fig = draw_ring(ring.nodes)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### MH ↔ MSS Assignment")
    assign_data = []
    for mss in ring.nodes:
        for mh in mss.mobile_hosts:
            assign_data.append({
                "Mobile Host": mh.id,
                "Connected to": f"MSS_{mss.id}",
                "Priority": mh.base_priority,
                "In CS?": "✅ Yes" if mh.in_cs else "No",
            })
    st.dataframe(pd.DataFrame(assign_data), use_container_width=True, hide_index=True)

# ────────── TAB 2: REQUEST PROPAGATION & GRANTING ──────────
with tab2:
    st.markdown("### 2 — Request Broadcasting & Priority-Based Granting")
    st.info(
        "**Replication mechanism:**\n"
        "1. MH sends request to its **local MSS**\n"
        "2. MSS **broadcasts** the request to **all other MSSs** (replication)\n"
        "3. Every MSS adds the request to its **replicated global queue**\n"
        "4. When token arrives at an MSS, it **grants** to the **highest-priority local MH**"
    )

    st.markdown("#### Broadcast Trace for Each Request")
    if st.session_state.reqs_made:
        for req in st.session_state.reqs_made:
            other_mss = [f"MSS_{m.id}" for m in ring.nodes if m.id != req.source_mss_id]
            with st.expander(f"📨 {req.request_id}  |  {req.mh_id} → MSS_{req.source_mss_id}  |  P={req.priority}  |  {req.status}"):
                st.write(f"**Origin:** {req.mh_id} at **MSS_{req.source_mss_id}**")
                st.write(f"**Priority:** {req.priority}  |  **Lamport Time:** {req.timestamp}")
                st.write(f"**Broadcast to:** {', '.join(other_mss)}")
                st.write(f"**Messages generated:** {len(other_mss)} (O(N−1) = {ring.n - 1})")
                st.write(f"**Current status:** `{req.status}`")
    else:
        st.warning("No requests yet. Use the sidebar to send requests.")

    st.markdown("---")
    st.markdown("#### Global Priority Queue (Token Holder's View)")
    holder = ring.token_holder()
    if holder and holder.global_queue:
        gq = []
        for rank, r in enumerate(holder.global_queue, 1):
            is_local = "✅" if r.source_mss_id == holder.id else ""
            gq.append({
                "Rank": rank,
                "MH": r.mh_id,
                "Source MSS": f"MSS_{r.source_mss_id}",
                "Priority": r.priority,
                "Lamport T": r.timestamp,
                "Local?": is_local,
                "Status": r.status,
            })
        st.dataframe(pd.DataFrame(gq), use_container_width=True, hide_index=True)
        st.caption(f"Token at **MSS_{holder.id}** — only **local** PENDING requests can be granted.")
    else:
        st.info("Global queue is empty or no token holder.")

    st.markdown("---")
    st.markdown("#### Event Log")
    if tm.event_log:
        for ev in tm.event_log[-15:]:
            st.text(ev)
    else:
        st.info("No events yet. Step the simulation.")

# ────────── TAB 3: REPLICATED LOGS & PRIORITIES ──────────
with tab3:
    st.markdown("### 3 — Replicated Request Logs at Each MSS")
    st.info(
        "Every MSS keeps a **complete replicated copy** of all requests.  "
        "This ensures any MSS can determine the correct priority order."
    )

    for mss in ring.nodes:
        token_badge = " 🔑 (Token Holder)" if mss.has_token else ""
        with st.expander(f"📋 MSS_{mss.id}{token_badge}  —  {len(mss.replicated_log)} log entries", expanded=True):
            if mss.replicated_log:
                log_data = []
                for idx, r in enumerate(mss.replicated_log, 1):
                    is_local = "Local" if r.source_mss_id == mss.id else "Replicated"
                    log_data.append({
                        "#": idx,
                        "Request ID": r.request_id,
                        "MH": r.mh_id,
                        "Origin MSS": f"MSS_{r.source_mss_id}",
                        "Priority": r.priority,
                        "Lamport T": r.timestamp,
                        "Type": is_local,
                        "Status": r.status,
                    })
                df = pd.DataFrame(log_data)
                # colour the status column
                st.dataframe(df, use_container_width=True, hide_index=True)
            else:
                st.caption("No entries yet")

    st.markdown("---")
    st.markdown("#### Verification: All MSSs Have Identical Request Sets")
    sets = []
    for mss in ring.nodes:
        ids = sorted(set(r.request_id for r in mss.replicated_log))
        sets.append(ids)
    if len(sets) > 1 and all(s == sets[0] for s in sets):
        st.success("✅ All MSSs have identical replicated request sets (replication is consistent)")
    elif not any(sets):
        st.info("No requests to compare yet")
    else:
        st.warning("⚠️ Logs not yet synchronized (requests may still be propagating)")

# ────────── TAB 4: QUEUES AFTER SERVICE ──────────
with tab4:
    st.markdown("### 4 — Queue State After Requests Have Been Served")
    st.info(
        "After a request is **granted** and later **completed**, it is removed from the "
        "global queue.  The tables below show the **current** queue state at every MSS."
    )

    cols = st.columns(ring.n)
    for idx, mss in enumerate(ring.nodes):
        with cols[idx]:
            tok = " 🔑" if mss.has_token else ""
            st.markdown(f"#### MSS_{mss.id}{tok}")

            # local queue
            st.caption("**Local Queue (own MHs)**")
            if mss.local_queue:
                ldf = pd.DataFrame([{
                    "MH": r.mh_id,
                    "P": r.priority,
                    "T": r.timestamp,
                    "Status": r.status,
                } for r in mss.local_queue])
                st.dataframe(ldf, hide_index=True, use_container_width=True)
            else:
                st.success("Empty ✓")

            # global priority queue
            st.caption("**Global Priority Queue**")
            if mss.global_queue:
                gdf = pd.DataFrame([{
                    "Rank": i + 1,
                    "MH": r.mh_id,
                    "MSS": r.source_mss_id,
                    "P": r.priority,
                    "T": r.timestamp,
                    "Status": r.status,
                } for i, r in enumerate(mss.global_queue)])
                st.dataframe(gdf, hide_index=True, use_container_width=True)
            else:
                st.success("Empty ✓")

    st.markdown("---")
    st.markdown("#### Completed Requests (Served & Removed from Queues)")
    completed = []
    seen_ids = set()
    for mss in ring.nodes:
        for r in mss.replicated_log:
            if r.status == "COMPLETED" and r.request_id not in seen_ids:
                completed.append(r.to_dict())
                seen_ids.add(r.request_id)
    if completed:
        st.dataframe(pd.DataFrame(completed), use_container_width=True, hide_index=True)
    else:
        st.warning("No completed requests yet.  Grant and complete some requests using the sidebar.")

    st.markdown("---")
    st.markdown("#### Per-MSS Statistics")
    st.dataframe(
        pd.DataFrame([mss.stats() for mss in ring.nodes]),
        use_container_width=True, hide_index=True,
    )

    # message complexity note
    total_msgs = sum(m.messages_sent + m.messages_received for m in ring.nodes)
    total_reqs = len(set(r.request_id for r in ring.nodes[0].replicated_log)) if ring.nodes[0].replicated_log else 0
    st.markdown("#### Message Complexity")
    st.info(
        f"**N (MSSs):** {ring.n}  \n"
        f"**Messages per broadcast:** N − 1 = {ring.n - 1}  \n"
        f"**Total unique requests:** {total_reqs}  \n"
        f"**Total messages exchanged:** {total_msgs}  \n"
        f"**Avg messages / request:** "
        f"{total_msgs / total_reqs:.1f}" if total_reqs else "—"
    )

# ── footer ──
st.markdown("---")
st.markdown(
    "<div style='text-align:center;color:gray;font-size:.85rem;'>"
    "<b>Token-Ring Mutual Exclusion with Replication</b> — "
    "MSS-MH Architecture · Lamport Clocks · Priority Queues · Request Broadcasting"
    "</div>",
    unsafe_allow_html=True,
)
