"""
Token-Ring Mutual Exclusion with Replication - Complete Streamlit App
=====================================================================
Demonstrates:
1. Animated token ring with MSSs and MHs (Fully Native Plotly JS Animation)
2. Request messages, token holding, handoff scenarios
3. Request broadcasting to all MSSs and priority-based granting
4. Replicated request logs and queue states
"""

import streamlit as st
import pandas as pd
import numpy as np
import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import plotly.graph_objects as go

# ═══════════════════════════════════════════════════════════════
#                      MODEL CLASSES
# ═══════════════════════════════════════════════════════════════

class LamportClock:
    def __init__(self): self.time = 0
    def increment(self): self.time += 1; return self.time
    def update(self, received_time): self.time = max(self.time, received_time) + 1; return self.time

@dataclass
class Request:
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
            "Request ID": self.request_id, "Mobile Host": self.mh_id,
            "Source MSS": f"MSS_{self.source_mss_id}", "Priority": self.priority,
            "Lamport Time": self.timestamp, "Status": self.status,
        }

class MobileHost:
    def __init__(self, mh_id: str, current_mss, base_priority: int = None):
        self.id = mh_id
        self.current_mss = current_mss
        self.base_priority = base_priority if base_priority else 5
        self.in_cs = False
        self.current_request: Optional[Request] = None

    def request_cs(self):
        if self.current_request and self.current_request.status == "PENDING": return None
        req = self.current_mss.receive_request_from_mh(self, self.base_priority)
        self.current_request = req
        return req

class MSS:
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
        req = Request(mh_id=mh.id, source_mss_id=self.id, priority=priority, timestamp=self.clock.time)
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
        if not self.has_token: return None
        local_pending = [r for r in self.global_queue if r.source_mss_id == self.id and r.status == "PENDING"]
        if not local_pending: return None
        granted = local_pending[0]
        granted.status = "GRANTED"
        if granted in self.local_queue: self.local_queue.remove(granted)
        self.grants_made += 1
        return granted

    def pass_token(self):
        if not self.has_token: return
        self.has_token = False
        self.next_mss.has_token = True
        self.next_mss.clock.increment()
        self.messages_sent += 1

    def stats(self):
        return {"MSS": f"MSS_{self.id}", "Sent": self.messages_sent, "Received": self.messages_received,
                "Grants": self.grants_made, "MHs": len(self.mobile_hosts),
                "Pending": len([r for r in self.local_queue if r.status == "PENDING"])}

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
            if m.has_token: return m
        return None

class TokenManager:
    def __init__(self, ring: RingTopology):
        self.ring = ring
        self.event_log: List[str] = []
    
    def step(self) -> Tuple[Optional[Request], str]:
        holder = self.ring.token_holder()
        if not holder: return None, "⚠️ No token holder found"
        granted = holder.grant_token()
        if granted:
            msg = f"✅ MSS_{holder.id} GRANTED token → {granted.mh_id}"
            self.event_log.append(msg)
            return granted, msg
        next_id = holder.next_mss.id
        holder.pass_token()
        msg = f"➡️ Token passed: MSS_{holder.id} → MSS_{next_id}"
        self.event_log.append(msg)
        return None, msg

# ═══════════════════════════════════════════════════════════════
#                   ROBUST PLOTLY NATIVE ANIMATION
# ═══════════════════════════════════════════════════════════════

class AnimationBuilder:
    """Builds a Plotly-native JavaScript animation. Constant trace count prevents flickering."""
    def __init__(self, num_mss: int, requesting_mh: str, handoff_mh: str):
        self.num_mss = num_mss
        self.req_mh = requesting_mh
        self.ho_mh = handoff_mh
        self.radius, self.mh_radius = 2.2, 0.55
        
        self.angles = [2 * math.pi * i / num_mss - math.pi / 2 for i in range(num_mss)]
        self.mss_x = [self.radius * math.cos(a) for a in self.angles]
        self.mss_y = [self.radius * math.sin(a) for a in self.angles]
        
        self.mh_pos = {}
        self.mh_names = {}
        for mss_id in range(num_mss):
            self.mh_pos[mss_id], self.mh_names[mss_id] = [], []
            for mh_idx in range(3):
                ang = self.angles[mss_id] + math.pi + (mh_idx - 1) * 0.4
                self.mh_pos[mss_id].append((self.mss_x[mss_id] + self.mh_radius * math.cos(ang),
                                            self.mss_y[mss_id] + self.mh_radius * math.sin(ang)))
                self.mh_names[mss_id].append(f"MH_{mss_id}_{chr(65 + mh_idx)}")
                
        self.req_mss_id = int(requesting_mh.split('_')[1])
        self.req_mh_idx = ord(requesting_mh.split('_')[2]) - 65
        self.ho_mss_id = int(handoff_mh.split('_')[1])
        self.ho_mh_idx = ord(handoff_mh.split('_')[2]) - 65
        self.ho_target_mss = (self.ho_mss_id + 1) % num_mss

        self.frames_data = []
        self.log_entries = []
        self.token_pos = 0.0  # Continuous float, never wraps back to 0 natively (prevents jumps)
        
        # Timing
        self.move_steps, self.stop_frames, self.hold_frames, self.msg_steps = 10, 8, 15, 8

    def lerp(self, a, b, t): return a + (b - a) * max(0.0, min(1.0, t))

    def build_frame(self, token_col, mss_cols, mh_cols, r_msg=None, p_msg=None, 
                    rl_msg=None, ho_line=None, log_text="", override_pos=None):
        """Must return exactly the same number of traces every time."""
        data = []
        c = {'ring': '#444', 'conn': '#666', 'nrm': '#00D4FF', 'req': '#FF5722'}
        
        # 1. Ring
        ra = np.linspace(0, 2 * np.pi, 100)
        data.append(go.Scatter(x=self.radius*1.1*np.cos(ra), y=self.radius*1.1*np.sin(ra), 
                               mode='lines', line=dict(color=c['ring'], width=3), hoverinfo='none'))
        
        # 2. Arrows
        for i in range(self.num_mss):
            mid = (self.angles[i] + self.angles[(i + 1) % self.num_mss]) / 2
            if i == self.num_mss - 1: mid = (self.angles[i] + self.angles[0] + 2 * math.pi) / 2
            ax, ay = self.radius*1.1*math.cos(mid), self.radius*1.1*math.sin(mid)
            data.append(go.Scatter(x=[ax, ax + 0.12*math.cos(mid+math.pi/2)], 
                                   y=[ay, ay + 0.12*math.sin(mid+math.pi/2)], 
                                   mode='lines', line=dict(color=c['ring'], width=2), hoverinfo='none'))
            
        # 3. Connection Lines
        lx, ly = [], []
        for mss_id in range(self.num_mss):
            for mh_idx in range(3):
                name = self.mh_names[mss_id][mh_idx]
                if override_pos and name in override_pos: continue
                lx.extend([self.mss_x[mss_id], self.mh_pos[mss_id][mh_idx][0], None])
                ly.extend([self.mss_y[mss_id], self.mh_pos[mss_id][mh_idx][1], None])
        data.append(go.Scatter(x=lx, y=ly, mode='lines', line=dict(color=c['conn'], width=1, dash='dot'), hoverinfo='none'))

        # 4. MSS Nodes
        mc = [mss_cols.get(i, c['nrm']) for i in range(self.num_mss)]
        ms = [55 if i in mss_cols else 48 for i in range(self.num_mss)]
        data.append(go.Scatter(x=self.mss_x, y=self.mss_y, mode='markers+text',
                               marker=dict(size=ms, color=mc, line=dict(width=3, color='white'), symbol='square'),
                               text=[f'MSS_{i}' for i in range(self.num_mss)], textposition='top center',
                               textfont=dict(color='white', size=11), hoverinfo='none'))

        # 5. MH Nodes
        hx, hy, hc, hs, ht = [], [], [], [], []
        for mss_id in range(self.num_mss):
            for mh_idx in range(3):
                name = self.mh_names[mss_id][mh_idx]
                mx, my = override_pos[name] if (override_pos and name in override_pos) else self.mh_pos[mss_id][mh_idx]
                hx.append(mx); hy.append(my); ht.append(name.split('_')[-1])
                hc.append(mh_cols.get(name, '#4CAF50')); hs.append(28 if name in mh_cols else 22)
        data.append(go.Scatter(x=hx, y=hy, mode='markers+text',
                               marker=dict(size=hs, color=hc, line=dict(width=2, color='white'), symbol='circle'),
                               text=ht, textposition='bottom center', textfont=dict(color='white', size=10), hoverinfo='none'))

        # Helpers for Messages
        def add_msg(msg_obj, text, color):
            if msg_obj: data.append(go.Scatter(x=[msg_obj['x']], y=[msg_obj['y']], opacity=1, mode='markers+text', text=[text], textposition='top center', textfont=dict(color='white', size=8), marker=dict(size=15, color=color, symbol='square', line=dict(width=2, color='white')), hoverinfo='none'))
            else: data.append(go.Scatter(x=[0], y=[0], opacity=0, hoverinfo='none'))

        # 6, 7, 8. Messages
        add_msg(r_msg, 'REQ', '#FF5722'); add_msg(p_msg, 'PERM', '#9C27B0'); add_msg(rl_msg, 'REL', '#2196F3')

        # 9. Handoff Line
        if ho_line: data.append(go.Scatter(x=[ho_line['x1'], ho_line['x2']], y=[ho_line['y1'], ho_line['y2']], mode='lines', line=dict(color='#FF9800', width=3, dash='dash'), hoverinfo='none'))
        else: data.append(go.Scatter(x=[0], y=[0], opacity=0, hoverinfo='none'))

        # 10. Token (Wrapped math applies here to keep loop continuous)
        tang = ((self.token_pos % self.num_mss) / self.num_mss) * 2 * math.pi - math.pi / 2
        data.append(go.Scatter(x=[self.radius*math.cos(tang)], y=[self.radius*math.sin(tang)], mode='markers+text',
                               marker=dict(size=35, color=token_col, symbol='circle', line=dict(width=4, color='#333')),
                               text=['🔑'], textfont=dict(size=14), hoverinfo='none'))

        # 11. Log text
        data.append(go.Scatter(x=[0], y=[-3.3], mode='text', text=[f'<b>{log_text}</b>'], textfont=dict(size=14, color='white'), hoverinfo='none'))
        return data

    def advance_token_to(self, target_node, **kw):
        """Calculates forward distance and adds frames without jumping backwards"""
        cur = self.token_pos % self.num_mss
        diff = (target_node - cur) % self.num_mss
        if diff < 0.01: return
        target_unwrapped = self.token_pos + diff
        start = self.token_pos
        for step in range(self.move_steps):
            self.token_pos = self.lerp(start, target_unwrapped, (step + 1) / self.move_steps)
            self.frames_data.append(self.build_frame(**kw))

    def hold(self, f_count, **kw):
        for _ in range(f_count): self.frames_data.append(self.build_frame(**kw))

    def msg(self, f_pos, t_pos, m_type, **kw):
        for step in range(self.msg_steps):
            t = (step + 1) / self.msg_steps
            kw[m_type] = {'x': self.lerp(f_pos[0], t_pos[0], t), 'y': self.lerp(f_pos[1], t_pos[1], t)}
            self.frames_data.append(self.build_frame(**kw))
            
    def generate(self):
        c = {'tf': '#FFF', 'th': '#0F0', 'mn': '#00D4FF', 'mp': '#FFD700', 'mh': '#0F0', 'mg': '#CC00FF', 'mr': '#FF5722', 'mc': '#9C27B0'}
        rx, ry = self.mh_pos[self.req_mss_id][self.req_mh_idx]
        smx, smy = self.mss_x[self.req_mss_id], self.mss_y[self.req_mss_id]
        hx, hy = self.mh_pos[self.ho_mss_id][self.ho_mh_idx]
        hmx, hmy = self.mss_x[self.ho_mss_id], self.mss_y[self.ho_mss_id]
        nx, ny = self.mh_pos[self.ho_target_mss][0]
        nmx, nmy = self.mss_x[self.ho_target_mss], self.mss_y[self.ho_target_mss]

        # 1. Request
        self.log_entries.append(f"1. {self.req_mh} requests CS")
        self.msg((rx, ry), (smx, smy), 'r_msg', token_col=c['tf'], mss_cols={}, mh_cols={self.req_mh: c['mr']}, log_text=f"📤 {self.req_mh} sending REQUEST to MSS_{self.req_mss_id}")
        self.hold(self.stop_frames, token_col=c['tf'], mss_cols={self.req_mss_id: c['mp']}, mh_cols={self.req_mh: c['mr']}, log_text=f"📋 MSS_{self.req_mss_id} queued request")

        # 2. Token circulation
        for node in range(1, self.req_mss_id + 1):
            self.advance_token_to(node, token_col=c['tf'], mss_cols={self.req_mss_id: c['mp']}, mh_cols={self.req_mh: c['mr']}, log_text=f"⚪ Token moving to MSS_{node}")
            if node != self.req_mss_id:
                self.hold(self.stop_frames, token_col=c['tf'], mss_cols={node: '#87CEEB', self.req_mss_id: c['mp']}, mh_cols={self.req_mh: c['mr']}, log_text=f"🔍 MSS_{node}: Checking queue → Empty")

        # 3. Granting
        self.log_entries.append("2. Token arrives, grants access")
        self.hold(self.hold_frames, token_col=c['th'], mss_cols={self.req_mss_id: c['mh']}, mh_cols={self.req_mh: c['mr']}, log_text=f"🟢 MSS_{self.req_mss_id} HOLDING token!")
        self.msg((smx, smy), (rx, ry), 'p_msg', token_col=c['th'], mss_cols={self.req_mss_id: c['mg']}, mh_cols={self.req_mh: c['mr']}, log_text=f"📨 Granting PERMISSION")
        self.hold(self.hold_frames, token_col=c['th'], mss_cols={self.req_mss_id: c['mh']}, mh_cols={self.req_mh: c['mc']}, log_text=f"🟣 {self.req_mh} in CRITICAL SECTION")
        
        # 4. Release & new request
        self.msg((rx, ry), (smx, smy), 'rl_msg', token_col=c['th'], mss_cols={self.req_mss_id: c['mh']}, mh_cols={self.req_mh: c['mc']}, log_text=f"📤 {self.req_mh} sending RELEASE")
        self.log_entries.append(f"3. Handoff Scenario: {self.ho_mh} requests, then moves")
        self.msg((hx, hy), (hmx, hmy), 'r_msg', token_col=c['tf'], mss_cols={}, mh_cols={self.ho_mh: c['mr']}, log_text=f"📤 {self.ho_mh} requesting at MSS_{self.ho_mss_id}")
        self.hold(self.stop_frames, token_col=c['tf'], mss_cols={self.ho_mss_id: c['mp']}, mh_cols={self.ho_mh: c['mr']}, log_text=f"📋 Request queued")

        # 5. Handoff movement
        for step in range(self.msg_steps * 2):
            t = (step + 1) / (self.msg_steps * 2)
            self.token_pos += 0.5 / (self.msg_steps * 2) # Slight token drift
            cx, cy = self.lerp(hx, nx, t), self.lerp(hy, ny, t)
            self.frames_data.append(self.build_frame(token_col=c['tf'], mss_cols={self.ho_mss_id: c['mp']}, mh_cols={self.ho_mh: '#FF9800'}, ho_line={'x1':hx,'y1':hy,'x2':cx,'y2':cy}, log_text=f"📱 HANDOFF: Moving to MSS_{self.ho_target_mss}…", override_pos={self.ho_mh:(cx,cy)}))

        # 6. Kill & Re-register
        for step in range(self.hold_frames):
            self.frames_data.append(self.build_frame(token_col=c['tf'], mss_cols={self.ho_mss_id: '#F00' if step%6<3 else c['mn']}, mh_cols={}, log_text=f"❌ Request KILLED at MSS_{self.ho_mss_id}", override_pos={self.ho_mh:(nx,ny)}))
        self.msg((nx, ny), (nmx, nmy), 'r_msg', token_col=c['tf'], mss_cols={}, mh_cols={self.ho_mh: c['mr']}, log_text=f"📤 RE-REGISTERING at MSS_{self.ho_target_mss}", override_pos={self.ho_mh:(nx,ny)})
        self.hold(self.stop_frames, token_col=c['tf'], mss_cols={self.ho_target_mss: c['mp']}, mh_cols={self.ho_mh: c['mr']}, log_text=f"📋 New MSS queued request", override_pos={self.ho_mh:(nx,ny)})

        # 7. Token arrives at new MSS
        self.log_entries.append("4. Token reaches new MSS and grants CS")
        cid = int(self.token_pos) + 1
        while cid % self.num_mss != self.ho_target_mss:
            tgt = cid % self.num_mss
            self.advance_token_to(tgt, token_col=c['tf'], mss_cols={self.ho_target_mss: c['mp']}, mh_cols={self.ho_mh: c['mr']}, log_text="⚪ Token passing...", override_pos={self.ho_mh:(nx,ny)})
            cid += 1
        self.advance_token_to(self.ho_target_mss, token_col=c['tf'], mss_cols={self.ho_target_mss: c['mp']}, mh_cols={self.ho_mh: c['mr']}, log_text="⚪ Token arriving...", override_pos={self.ho_mh:(nx,ny)})
        
        self.hold(self.hold_frames, token_col=c['th'], mss_cols={self.ho_target_mss: c['mh']}, mh_cols={self.ho_mh: c['mr']}, log_text="🟢 HOLDING token!", override_pos={self.ho_mh:(nx,ny)})
        self.msg((nmx, nmy), (nx, ny), 'p_msg', token_col=c['th'], mss_cols={self.ho_target_mss: c['mg']}, mh_cols={self.ho_mh: c['mr']}, log_text="📨 Granting PERMISSION", override_pos={self.ho_mh:(nx,ny)})
        self.hold(self.hold_frames, token_col=c['th'], mss_cols={self.ho_target_mss: c['mh']}, mh_cols={self.ho_mh: c['mc']}, log_text=f"🟣 {self.ho_mh} in CRITICAL SECTION", override_pos={self.ho_mh:(nx,ny)})

        # Wrap up traces into Native Plotly format
        frames = [go.Frame(data=f, name=str(i)) for i, f in enumerate(self.frames_data)]
        
        # Native Playback UI
        layout = go.Layout(
            title=dict(text='<b>Native Token Ring Animation (No Flickering!)</b>', font=dict(color='white', size=18), x=0.5),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-3.5, 3.5]),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-3.8, 3.5], scaleanchor='x'),
            plot_bgcolor='#1a1a2e', paper_bgcolor='#1a1a2e', height=750, margin=dict(l=20, r=20, t=60, b=20),
            showlegend=False,
            updatemenus=[dict(
                type='buttons', showactive=False, x=0.05, y=-0.05, xanchor='left', yanchor='top', direction='right',
                buttons=[
                    dict(label='▶ Play', method='animate', args=[None, dict(frame=dict(duration=50, redraw=True), fromcurrent=True, transition=dict(duration=0))]),
                    dict(label='⏸ Pause', method='animate', args=[[None], dict(frame=dict(duration=0, redraw=False), mode='immediate', transition=dict(duration=0))])
                ]
            )],
            sliders=[dict(
                active=0, yanchor='top', xanchor='left', currentvalue=dict(font=dict(size=12, color='white'), prefix='Frame: ', visible=True, xanchor='right'),
                transition=dict(duration=0), pad=dict(b=10, t=30), len=0.8, x=0.2, y=-0.05,
                steps=[dict(args=[[str(i)], dict(frame=dict(duration=0, redraw=True), mode='immediate', transition=dict(duration=0))], label='', method='animate') for i in range(len(frames))]
            )]
        )
        return self.frames_data[0], frames, layout, self.log_entries

# ═══════════════════════════════════════════════════════════════
#                       STREAMLIT APP
# ═══════════════════════════════════════════════════════════════

st.set_page_config(page_title='Token-Ring ME', page_icon='🔐', layout='wide')
st.markdown("""<style>.block-container{padding-top:1rem;}
.header{font-size:2rem;font-weight:700;color:#0d47a1;text-align:center;padding:.8rem;background:linear-gradient(90deg,#e3f2fd,#bbdefb);border-radius:10px;margin-bottom:1.2rem;}</style>
<div class="header">🔐 Token-Ring Mutual Exclusion — Replication Scheme</div>""", unsafe_allow_html=True)

if 'ring' not in st.session_state:
    r = RingTopology(4); m = []
    for i in range(4):
        for j in range(3):
            mh = MobileHost(f"MH_{i}_{chr(65+j)}", r.nodes[i], 5+i+j); r.nodes[i].add_mh(mh); m.append(mh)
    st.session_state.ring = r; st.session_state.mhs = m; st.session_state.tm = TokenManager(r)

with st.sidebar:
    st.header('⚙️ Controls')
    if st.button('🔄 Reset State', use_container_width=True):
        for k in list(st.session_state.keys()): del st.session_state[k]
        st.rerun()

tab1, tab2 = st.tabs(['🎬 1. Animation (Smooth JS)', '📊 2. Logic State'])

with tab1:
    st.markdown('### Native Plotly Animation Demo')
    st.info('✨ **Fixed:** Uses pure browser JS (No Streamlit loop flickering) and absolute mathematical tracking (No Token jumps)! Click **Play** on the chart below.')
    
    col1, col2, col3 = st.columns(3)
    num_mss_anim = col1.selectbox('Number of MSSs', [4, 5, 6], index=2)
    mh_opts = [f"MH_{i}_{chr(65+j)}" for i in range(num_mss_anim) for j in range(3)]
    req_mh = col2.selectbox('Requesting MH', mh_opts, index=min(6, len(mh_opts)-1))
    ho_mh = col3.selectbox('Handoff MH', [m for m in mh_opts if m != req_mh], index=4)

    if st.button('🎬 Generate Animation Sequence', type='primary'):
        with st.spinner('Compiling Javascript Frames...'):
            builder = AnimationBuilder(num_mss_anim, req_mh, ho_mh)
            init_data, frames, layout, log = builder.generate()
            st.session_state.anim = (init_data, frames, layout, log)

    if 'anim' in st.session_state:
        init_data, frames, layout, log = st.session_state.anim
        
        # Single chart render, Plotly UI handles the rest perfectly!
        fig = go.Figure(data=init_data, frames=frames, layout=layout)
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown('#### Event Log')
        for i, entry in enumerate(log, 1): st.caption(f"{i}. {entry}")

with tab2:
    st.markdown('### Current Ring State & Queues')
    cols = st.columns(st.session_state.ring.n)
    for i, mss in enumerate(st.session_state.ring.nodes):
        with cols[i]:
            st.markdown(f"**MSS_{mss.id}** {' 🔑' if mss.has_token else ''}")
            st.dataframe(pd.DataFrame([{'MH': r.mh_id, 'Status': r.status} for r in mss.local_queue]) if mss.local_queue else pd.DataFrame(), hide_index=True)
