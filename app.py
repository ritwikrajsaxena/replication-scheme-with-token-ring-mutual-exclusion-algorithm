"""
Token-Ring Mutual Exclusion with Replication - Complete Streamlit App
=====================================================================
Demonstrates:
1. Animated token ring showing message transmission cycle
2. Request broadcasting to all MSSs and priority-based granting
3. Replicated request logs at each MSS with priorities
4. Queue state after requests have been served
"""

import streamlit as st
import pandas as pd
import numpy as np
import math
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import plotly.graph_objects as go

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
        self.base_priority = base_priority if base_priority else 5
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
#                   ANIMATED TOKEN RING VISUALIZATION
# ═══════════════════════════════════════════════════════════════

def create_animated_token_ring(num_mss: int = 6, source_mss: int = 1, dest_mss: int = 4):
    """
    Create an animated Plotly figure showing token ring operation.
    
    Phases:
    1. IDLE: Free token (white) circulates
    2. LOADING: Token stops at source, turns yellow
    3. TRANSMITTING: Token (green) moves to destination
    4. READING: Token at destination, turns purple
    5. RETURNING: Token (blue) returns to source
    6. FINISHING: Token back at source, turns white again
    """
    
    radius = 2.0
    angles = [2 * math.pi * i / num_mss - math.pi / 2 for i in range(num_mss)]
    mss_x = [radius * math.cos(a) for a in angles]
    mss_y = [radius * math.sin(a) for a in angles]
    
    # Color scheme
    colors = {
        'idle': '#FFFFFF',       # White - free token
        'loading': '#FFD700',    # Yellow - loading data
        'transmitting': '#00FF00', # Green - carrying message
        'reading': '#CC00FF',    # Purple - being read
        'returning': '#00BBFF',  # Blue - carrying receipt
        'finishing': '#FFFFFF',  # White - back to free
    }
    
    # Generate animation frames
    frames = []
    frame_data = []  # Store frame info for logs
    
    # Parameters
    steps_per_node = 10  # Steps to move between nodes
    wait_steps = 15      # Steps to wait at a node
    
    token_pos = 0.0  # Start at MSS 0
    frame_idx = 0
    
    # Helper function to create a frame
    def make_frame(token_angle, token_color, phase, current_node=None, frame_num=0):
        tx = radius * math.cos(token_angle)
        ty = radius * math.sin(token_angle)
        
        # MSS nodes
        mss_colors = ['#00D4FF'] * num_mss
        mss_sizes = [40] * num_mss
        
        # Highlight active nodes
        if current_node is not None:
            if phase in ['loading', 'finishing']:
                mss_colors[current_node] = '#FFD700'  # Yellow
                mss_sizes[current_node] = 50
            elif phase == 'reading':
                mss_colors[current_node] = '#CC00FF'  # Purple
                mss_sizes[current_node] = 50
        
        # Create frame data
        data = [
            # Ring
            go.Scatter(
                x=[radius * 1.05 * math.cos(a) for a in np.linspace(0, 2*math.pi, 100)],
                y=[radius * 1.05 * math.sin(a) for a in np.linspace(0, 2*math.pi, 100)],
                mode='lines',
                line=dict(color='#444444', width=2),
                hoverinfo='none',
                showlegend=False,
            ),
            # MSS Nodes
            go.Scatter(
                x=mss_x,
                y=mss_y,
                mode='markers+text',
                marker=dict(size=mss_sizes, color=mss_colors, 
                           line=dict(width=2, color='white')),
                text=[f'MSS_{i}' for i in range(num_mss)],
                textposition='top center',
                textfont=dict(color='white', size=12),
                hoverinfo='text',
                hovertext=[f'MSS_{i}' for i in range(num_mss)],
                showlegend=False,
            ),
            # Token
            go.Scatter(
                x=[tx],
                y=[ty],
                mode='markers',
                marker=dict(size=25, color=token_color,
                           line=dict(width=3, color='black'),
                           symbol='circle'),
                hoverinfo='text',
                hovertext=f'Token ({phase.upper()})',
                showlegend=False,
            ),
        ]
        
        return go.Frame(data=data, name=str(frame_num))
    
    # Phase 1: IDLE - Token circulates from MSS 0 towards source
    phase = 'idle'
    for node in range(source_mss + 1):
        for step in range(steps_per_node):
            progress = node + step / steps_per_node
            angle = (progress / num_mss) * 2 * math.pi - math.pi / 2
            frames.append(make_frame(angle, colors['idle'], phase, None, frame_idx))
            frame_data.append({'phase': 'IDLE', 'desc': f'Free token circulating... Position: {progress:.1f}'})
            frame_idx += 1
    
    # Phase 2: LOADING - Token stops at source MSS
    phase = 'loading'
    angle = (source_mss / num_mss) * 2 * math.pi - math.pi / 2
    for step in range(wait_steps):
        frames.append(make_frame(angle, colors['loading'], phase, source_mss, frame_idx))
        frame_data.append({'phase': 'LOADING', 'desc': f'MSS_{source_mss} capturing token, loading message for MSS_{dest_mss}'})
        frame_idx += 1
    
    # Phase 3: TRANSMITTING - Token moves to destination
    phase = 'transmitting'
    start_node = source_mss
    end_node = dest_mss if dest_mss > source_mss else dest_mss + num_mss
    
    for node_offset in range(end_node - start_node + 1):
        current_node = (start_node + node_offset) % num_mss
        for step in range(steps_per_node):
            progress = start_node + node_offset + step / steps_per_node
            angle = (progress / num_mss) * 2 * math.pi - math.pi / 2
            frames.append(make_frame(angle, colors['transmitting'], phase, None, frame_idx))
            frame_data.append({'phase': 'TRANSMITTING', 'desc': f'Token carrying message to MSS_{dest_mss}...'})
            frame_idx += 1
    
    # Phase 4: READING - Token at destination
    phase = 'reading'
    angle = (dest_mss / num_mss) * 2 * math.pi - math.pi / 2
    for step in range(wait_steps):
        frames.append(make_frame(angle, colors['reading'], phase, dest_mss, frame_idx))
        frame_data.append({'phase': 'READING', 'desc': f'MSS_{dest_mss} reading message, preparing acknowledgment'})
        frame_idx += 1
    
    # Phase 5: RETURNING - Token returns to source
    phase = 'returning'
    start_node = dest_mss
    end_node = source_mss if source_mss > dest_mss else source_mss + num_mss
    
    for node_offset in range(end_node - start_node + 1):
        current_node = (start_node + node_offset) % num_mss
        for step in range(steps_per_node):
            progress = start_node + node_offset + step / steps_per_node
            angle = (progress / num_mss) * 2 * math.pi - math.pi / 2
            frames.append(make_frame(angle, colors['returning'], phase, None, frame_idx))
            frame_data.append({'phase': 'RETURNING', 'desc': f'Token carrying acknowledgment back to MSS_{source_mss}...'})
            frame_idx += 1
    
    # Phase 6: FINISHING - Token back at source
    phase = 'finishing'
    angle = (source_mss / num_mss) * 2 * math.pi - math.pi / 2
    for step in range(wait_steps):
        frames.append(make_frame(angle, colors['finishing'], phase, source_mss, frame_idx))
        frame_data.append({'phase': 'FINISHING', 'desc': f'MSS_{source_mss} received acknowledgment, releasing free token'})
        frame_idx += 1
    
    # Continue as free token
    phase = 'idle'
    for node_offset in range(num_mss):
        current_node = (source_mss + node_offset) % num_mss
        for step in range(steps_per_node):
            progress = source_mss + node_offset + step / steps_per_node
            angle = (progress / num_mss) * 2 * math.pi - math.pi / 2
            frames.append(make_frame(angle, colors['idle'], phase, None, frame_idx))
            frame_data.append({'phase': 'IDLE', 'desc': 'Free token circulating, ready for next request'})
            frame_idx += 1
    
    # Create initial figure
    initial_angle = -math.pi / 2
    initial_tx = radius * math.cos(initial_angle)
    initial_ty = radius * math.sin(initial_angle)
    
    fig = go.Figure(
        data=[
            # Ring
            go.Scatter(
                x=[radius * 1.05 * math.cos(a) for a in np.linspace(0, 2*math.pi, 100)],
                y=[radius * 1.05 * math.sin(a) for a in np.linspace(0, 2*math.pi, 100)],
                mode='lines',
                line=dict(color='#444444', width=2),
                hoverinfo='none',
                showlegend=False,
            ),
            # MSS Nodes
            go.Scatter(
                x=mss_x,
                y=mss_y,
                mode='markers+text',
                marker=dict(size=40, color='#00D4FF', 
                           line=dict(width=2, color='white')),
                text=[f'MSS_{i}' for i in range(num_mss)],
                textposition='top center',
                textfont=dict(color='white', size=12),
                hoverinfo='text',
                hovertext=[f'MSS_{i}' for i in range(num_mss)],
                showlegend=False,
            ),
            # Token
            go.Scatter(
                x=[initial_tx],
                y=[initial_ty],
                mode='markers',
                marker=dict(size=25, color='#FFFFFF',
                           line=dict(width=3, color='black'),
                           symbol='circle'),
                hoverinfo='text',
                hovertext='Token (IDLE)',
                showlegend=False,
            ),
        ],
        frames=frames,
        layout=go.Layout(
            title=dict(
                text=f'<b>Token Ring Animation</b><br><sup>MSS_{source_mss} → MSS_{dest_mss}</sup>',
                font=dict(color='white', size=16),
            ),
            xaxis=dict(
                showgrid=False, zeroline=False, showticklabels=False,
                range=[-3, 3],
            ),
            yaxis=dict(
                showgrid=False, zeroline=False, showticklabels=False,
                range=[-3, 3],
                scaleanchor='x',
            ),
            plot_bgcolor='#1a1a2e',
            paper_bgcolor='#1a1a2e',
            height=500,
            margin=dict(l=20, r=20, t=60, b=20),
            updatemenus=[
                dict(
                    type='buttons',
                    showactive=False,
                    y=1.15,
                    x=0.5,
                    xanchor='center',
                    buttons=[
                        dict(
                            label='▶ Play',
                            method='animate',
                            args=[
                                None,
                                dict(
                                    frame=dict(duration=50, redraw=True),
                                    fromcurrent=True,
                                    mode='immediate',
                                )
                            ]
                        ),
                        dict(
                            label='⏸ Pause',
                            method='animate',
                            args=[
                                [None],
                                dict(
                                    frame=dict(duration=0, redraw=False),
                                    mode='immediate',
                                )
                            ]
                        ),
                        dict(
                            label='🔄 Reset',
                            method='animate',
                            args=[
                                [str(0)],
                                dict(
                                    frame=dict(duration=0, redraw=True),
                                    mode='immediate',
                                )
                            ]
                        ),
                    ]
                )
            ],
            sliders=[
                dict(
                    active=0,
                    yanchor='top',
                    xanchor='left',
                    currentvalue=dict(
                        font=dict(size=12, color='white'),
                        prefix='Frame: ',
                        visible=True,
                        xanchor='right'
                    ),
                    pad=dict(b=10, t=50),
                    len=0.9,
                    x=0.05,
                    y=0,
                    steps=[
                        dict(
                            args=[[str(i)], dict(frame=dict(duration=50, redraw=True), mode='immediate')],
                            label=str(i),
                            method='animate'
                        )
                        for i in range(0, len(frames), 10)  # Show every 10th frame in slider
                    ]
                )
            ]
        )
    )
    
    return fig, frame_data


def create_static_ring_with_token(mss_list, token_pos=0, token_color='#FFFFFF', phase='IDLE'):
    """Create a static ring visualization for the DME simulation tabs"""
    n = len(mss_list)
    radius = 2.0
    angles = [2 * math.pi * i / n - math.pi / 2 for i in range(n)]
    
    xs = [radius * math.cos(a) for a in angles]
    ys = [radius * math.sin(a) for a in angles]
    
    fig = go.Figure()
    
    # Ring circle
    ring_angles = np.linspace(0, 2 * math.pi, 100)
    fig.add_trace(go.Scatter(
        x=[radius * 1.05 * math.cos(a) for a in ring_angles],
        y=[radius * 1.05 * math.sin(a) for a in ring_angles],
        mode='lines',
        line=dict(color='#555', width=2),
        hoverinfo='none',
        showlegend=False,
    ))
    
    # Arrows between nodes
    for i in range(n):
        x0, y0 = xs[i], ys[i]
        x1, y1 = xs[(i + 1) % n], ys[(i + 1) % n]
        
        dx, dy = x1 - x0, y1 - y0
        length = math.hypot(dx, dy)
        if length > 0:
            shrink = 0.25
            lx0 = x0 + dx * shrink
            ly0 = y0 + dy * shrink
            lx1 = x1 - dx * shrink
            ly1 = y1 - dy * shrink
            
            fig.add_trace(go.Scatter(
                x=[lx0, lx1], y=[ly0, ly1],
                mode='lines',
                line=dict(width=2, color='#555'),
                hoverinfo='none',
                showlegend=False,
            ))
            
            # Arrowhead
            arrow_size = 0.12
            udx, udy = dx / length, dy / length
            px, py = -udy, udx
            tip_x, tip_y = lx1, ly1
            base_x1 = tip_x - udx * arrow_size + px * arrow_size * 0.4
            base_y1 = tip_y - udy * arrow_size + py * arrow_size * 0.4
            base_x2 = tip_x - udx * arrow_size - px * arrow_size * 0.4
            base_y2 = tip_y - udy * arrow_size - py * arrow_size * 0.4
            
            fig.add_trace(go.Scatter(
                x=[tip_x, base_x1, base_x2, tip_x],
                y=[tip_y, base_y1, base_y2, tip_y],
                fill='toself',
                fillcolor='#555',
                line=dict(width=0),
                hoverinfo='none',
                showlegend=False,
            ))
    
    # MSS nodes
    colors = ['gold' if mss.has_token else '#64b5f6' for mss in mss_list]
    sizes = [55 if mss.has_token else 42 for mss in mss_list]
    labels = []
    hovers = []
    
    for mss in mss_list:
        tok = ' 🔑' if mss.has_token else ''
        labels.append(f'MSS_{mss.id}{tok}')
        mh_names = ', '.join(f'{mh.id}(P{mh.base_priority})' for mh in mss.mobile_hosts) or '—'
        hovers.append(f'<b>MSS_{mss.id}</b>{tok}<br>MHs: {mh_names}<br>Pending: {len(mss.local_queue)}')
    
    fig.add_trace(go.Scatter(
        x=xs, y=ys,
        mode='markers+text',
        text=labels,
        textposition='top center',
        textfont=dict(size=13, color='black'),
        hovertext=hovers,
        hoverinfo='text',
        marker=dict(size=sizes, color=colors, line=dict(width=2, color='black')),
        showlegend=False,
    ))
    
    # MH labels
    for idx, mss in enumerate(mss_list):
        for j, mh in enumerate(mss.mobile_hosts):
            offset_angle = angles[idx] + math.pi + (j - len(mss.mobile_hosts) / 2) * 0.35
            mx = xs[idx] + 0.6 * math.cos(offset_angle)
            my = ys[idx] + 0.6 * math.sin(offset_angle)
            
            fig.add_trace(go.Scatter(
                x=[xs[idx], mx], y=[ys[idx], my],
                mode='lines',
                line=dict(width=1, color='#aaa', dash='dot'),
                hoverinfo='none',
                showlegend=False,
            ))
            
            fig.add_trace(go.Scatter(
                x=[mx], y=[my],
                mode='markers+text',
                text=[f'📱{mh.id} P={mh.base_priority}'],
                textposition='bottom center',
                textfont=dict(size=9, color='#333'),
                marker=dict(size=12, color='#e8f5e9', line=dict(width=1, color='#66bb6a')),
                hoverinfo='text',
                hovertext=f'{mh.id} | Priority={mh.base_priority} | MSS_{mss.id}',
                showlegend=False,
            ))
    
    fig.update_layout(
        title=dict(text='<b>MSS-MH Token Ring</b>', font=dict(size=18)),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-3.5, 3.5]),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-3.5, 3.5], scaleanchor='x'),
        height=600,
        plot_bgcolor='white',
        paper_bgcolor='white',
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
        (0, 'MH_A', 5),
        (0, 'MH_B', 3),
        (1, 'MH_C', 8),
        (1, 'MH_D', 6),
        (2, 'MH_E', 4),
        (2, 'MH_F', 7),
        (3, 'MH_G', 2),
        (3, 'MH_H', 9),
    ]
    for mss_id, mh_id, pri in config:
        mh = MobileHost(mh_id, ring.nodes[mss_id], base_priority=pri)
        ring.nodes[mss_id].add_mh(mh)
        mhs.append(mh)
    return ring, mhs


# ═══════════════════════════════════════════════════════════════
#                       STREAMLIT APP
# ═══════════════════════════════════════════════════════════════

st.set_page_config(page_title='Token-Ring ME Replication', page_icon='🔐', layout='wide')

st.markdown(
    """
    <style>
    .block-container{padding-top:1rem;}
    .header{font-size:2rem;font-weight:700;color:#0d47a1;text-align:center;
            padding:.8rem;background:linear-gradient(90deg,#e3f2fd,#bbdefb);
            border-radius:10px;margin-bottom:1.2rem;}
    .sub{color:#555;text-align:center;margin-bottom:1.5rem;}
    .phase-box{padding:1rem;border-radius:8px;margin:0.5rem 0;}
    .phase-idle{background-color:#f5f5f5;border-left:4px solid #9e9e9e;}
    .phase-loading{background-color:#fff8e1;border-left:4px solid #ffc107;}
    .phase-transmitting{background-color:#e8f5e9;border-left:4px solid #4caf50;}
    .phase-reading{background-color:#f3e5f5;border-left:4px solid #9c27b0;}
    .phase-returning{background-color:#e3f2fd;border-left:4px solid #2196f3;}
    .phase-finishing{background-color:#fff8e1;border-left:4px solid #ff9800;}
    </style>
    <div class="header">🔐 Token-Ring Mutual Exclusion — Replication Scheme</div>
    <p class="sub">MSS-MH Architecture &nbsp;|&nbsp; Request Broadcasting &nbsp;|&nbsp;
    Priority-Based Granting &nbsp;|&nbsp; Replicated Logs &amp; Queues</p>
    """,
    unsafe_allow_html=True,
)

# Session state initialization
if 'ring' not in st.session_state:
    r, m = build_default_scenario()
    st.session_state.ring = r
    st.session_state.mhs = m
    st.session_state.tm = TokenManager(r)
    st.session_state.step = 0
    st.session_state.reqs_made: List[Request] = []

ring: RingTopology = st.session_state.ring
mhs: List[MobileHost] = st.session_state.mhs
tm: TokenManager = st.session_state.tm

# Sidebar
with st.sidebar:
    st.header('⚙️ Controls')

    if st.button('🔄 Reset Everything'):
        r, m = build_default_scenario()
        st.session_state.ring = r
        st.session_state.mhs = m
        st.session_state.tm = TokenManager(r)
        st.session_state.step = 0
        st.session_state.reqs_made = []
        st.rerun()

    st.markdown('---')
    st.subheader('📤 Send Request')
    opts = [f'{mh.id}  (at MSS_{mh.current_mss.id}, P={mh.base_priority})' for mh in mhs]
    sel = st.selectbox('Select Mobile Host', range(len(mhs)), format_func=lambda i: opts[i])
    if st.button('Send Request', use_container_width=True):
        req = mhs[sel].request_cs()
        if req:
            st.session_state.reqs_made.append(req)
            st.success(f'{mhs[sel].id} requested CS')
        else:
            st.warning('Already has a pending request')

    st.markdown('---')
    st.subheader('🔄 Token Circulation')

    c1, c2 = st.columns(2)
    with c1:
        if st.button('▶ Step', use_container_width=True):
            granted, ev = tm.step()
            st.session_state.step += 1
            if granted:
                st.success(f'Granted → {granted.mh_id}')
    with c2:
        if st.button('⏩ ×5', use_container_width=True):
            for _ in range(5):
                tm.step()
                st.session_state.step += 1

    st.markdown('---')
    st.subheader('🏁 Complete CS')
    granted_list = [r for r in st.session_state.reqs_made if r.status == 'GRANTED']
    if granted_list:
        g_opts = [f'{r.mh_id} ({r.request_id})' for r in granted_list]
        g_sel = st.selectbox('Select granted request', range(len(granted_list)), format_func=lambda i: g_opts[i])
        if st.button('Mark Completed', use_container_width=True):
            chosen = granted_list[g_sel]
            tm.complete(chosen)
            for mh in mhs:
                if mh.id == chosen.mh_id:
                    mh.exit_cs()
            st.success(f'{chosen.mh_id} completed CS')
    else:
        st.info('No granted requests to complete')

    st.markdown('---')
    holder = ring.token_holder()
    st.metric('Steps', st.session_state.step)
    st.metric('Token At', f'MSS_{holder.id}' if holder else '—')
    st.metric('Circulations', tm.circulations)


# ═══════════════════════════════════════════════════════════════
#                           TABS
# ═══════════════════════════════════════════════════════════════

tab1, tab2, tab3, tab4 = st.tabs([
    '🔗 1. Ring Topology & Animation',
    '📡 2. Request Propagation & Granting',
    '📋 3. Request Logs & Priorities',
    '📊 4. Queues After Service',
])

# ────────── TAB 1: ANIMATED RING TOPOLOGY ──────────
with tab1:
    st.markdown('### 1 — Token Ring Animation: Complete Transmission Cycle')
    
    st.info('''
    **📺 Animation Demo: MSS_1 sends message to MSS_4**
    
    Watch the complete token lifecycle:
    1. ⚪ **FREE TOKEN** (White) — Token circulates looking for requests
    2. 🟡 **LOADING** (Yellow) — MSS_1 captures token, loads message
    3. 🟢 **TRANSMITTING** (Green) — Token carries data to MSS_4
    4. 🟣 **READING** (Purple) — MSS_4 reads message, prepares ACK
    5. 🔵 **RETURNING** (Blue) — Token carries acknowledgment back
    6. ⚪ **FINISHING** (White) — MSS_1 receives ACK, releases free token
    
    **Click ▶ Play to start the animation!**
    ''')
    
    # Create animated visualization
    source_mss = 1
    dest_mss = 4
    
    anim_fig, frame_data = create_animated_token_ring(
        num_mss=6, 
        source_mss=source_mss, 
        dest_mss=dest_mss
    )
    
    st.plotly_chart(anim_fig, use_container_width=True)
    
    # Token color legend
    st.markdown('#### 🎨 Token Color Legend')
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.markdown('''
        <div style="background:#FFFFFF;color:black;padding:10px;border-radius:5px;text-align:center;border:2px solid #333;">
        ⚪ IDLE<br><small>Free Token</small>
        </div>
        ''', unsafe_allow_html=True)
    with col2:
        st.markdown('''
        <div style="background:#FFD700;color:black;padding:10px;border-radius:5px;text-align:center;">
        🟡 LOADING<br><small>Capturing</small>
        </div>
        ''', unsafe_allow_html=True)
    with col3:
        st.markdown('''
        <div style="background:#00FF00;color:black;padding:10px;border-radius:5px;text-align:center;">
        🟢 TRANSMIT<br><small>Carrying Data</small>
        </div>
        ''', unsafe_allow_html=True)
    with col4:
        st.markdown('''
        <div style="background:#CC00FF;color:white;padding:10px;border-radius:5px;text-align:center;">
        🟣 READING<br><small>At Destination</small>
        </div>
        ''', unsafe_allow_html=True)
    with col5:
        st.markdown('''
        <div style="background:#00BBFF;color:black;padding:10px;border-radius:5px;text-align:center;">
        🔵 RETURN<br><small>Carrying ACK</small>
        </div>
        ''', unsafe_allow_html=True)
    
    st.markdown('---')
    
    # Step-by-step explanation
    st.markdown('#### 📜 Step-by-Step Transmission Log')
    
    log_phases = [
        ('IDLE', '⚪', 'Free token circulates on the ring, passing through MSS_0...', '#f5f5f5'),
        ('IDLE', '⚪', 'Token arrives at MSS_1 which has a pending request for MSS_4', '#f5f5f5'),
        ('LOADING', '🟡', 'MSS_1 CAPTURES the token (Yellow flash)', '#fff8e1'),
        ('LOADING', '🟡', 'MSS_1 writes message data and destination address (MSS_4) onto token', '#fff8e1'),
        ('TRANSMITTING', '🟢', 'Token (now Green) departs MSS_1 carrying the message', '#e8f5e9'),
        ('TRANSMITTING', '🟢', 'Token passes MSS_2 — address doesn\'t match, forwarded', '#e8f5e9'),
        ('TRANSMITTING', '🟢', 'Token passes MSS_3 — address doesn\'t match, forwarded', '#e8f5e9'),
        ('TRANSMITTING', '🟢', 'Token arrives at MSS_4 — DESTINATION REACHED!', '#e8f5e9'),
        ('READING', '🟣', 'MSS_4 READS the message (Purple flash)', '#f3e5f5'),
        ('READING', '🟣', 'MSS_4 copies data to memory and marks acknowledgment on token', '#f3e5f5'),
        ('RETURNING', '🔵', 'Token (now Blue with ACK) departs MSS_4', '#e3f2fd'),
        ('RETURNING', '🔵', 'Token passes MSS_5 — not the sender, forwarded', '#e3f2fd'),
        ('RETURNING', '🔵', 'Token passes MSS_0 — not the sender, forwarded', '#e3f2fd'),
        ('RETURNING', '🔵', 'Token arrives back at MSS_1 — SENDER REACHED!', '#e3f2fd'),
        ('FINISHING', '🟡', 'MSS_1 reads the acknowledgment (Purple flash)', '#fff8e1'),
        ('FINISHING', '⚪', 'MSS_1 confirms delivery and RELEASES free token', '#f5f5f5'),
        ('IDLE', '⚪', 'Free token continues circulating for next request...', '#f5f5f5'),
    ]
    
    for phase, icon, desc, color in log_phases:
        st.markdown(f'''
        <div style="background:{color};padding:8px 12px;border-radius:5px;margin:4px 0;
                    border-left:4px solid {'#ffc107' if phase=='LOADING' else '#4caf50' if phase=='TRANSMITTING' 
                    else '#9c27b0' if phase=='READING' else '#2196f3' if phase=='RETURNING' 
                    else '#ff9800' if phase=='FINISHING' else '#9e9e9e'};">
        <b>{icon} [{phase}]</b> {desc}
        </div>
        ''', unsafe_allow_html=True)
    
    st.markdown('---')
    st.markdown('#### 🔄 Why This Circular Design Matters')
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown('''
        **⚖️ Fairness**
        
        Every MSS gets an equal turn to capture the token. No starvation possible.
        ''')
    with col2:
        st.markdown('''
        **🚫 No Collisions**
        
        Only ONE active token exists. Two stations can never crash into each other's signals.
        ''')
    with col3:
        st.markdown('''
        **✅ Accountability**
        
        Sender always gets a receipt back, confirming the data arrived successfully.
        ''')


# ────────── TAB 2: REQUEST PROPAGATION & GRANTING ──────────
with tab2:
    st.markdown('### 2 — Request Broadcasting & Priority-Based Granting')
    
    st.info('''
    **Replication mechanism:**
    1. MH sends request to its **local MSS**
    2. MSS **broadcasts** the request to **all other MSSs** (replication)
    3. Every MSS adds the request to its **replicated global queue**
    4. When token arrives at an MSS, it **grants** to the **highest-priority local MH**
    ''')
    
    # Show the static ring with current state
    static_fig = create_static_ring_with_token(ring.nodes)
    st.plotly_chart(static_fig, use_container_width=True)
    
    st.markdown('#### Broadcast Trace for Each Request')
    if st.session_state.reqs_made:
        for req in st.session_state.reqs_made:
            other_mss = [f'MSS_{m.id}' for m in ring.nodes if m.id != req.source_mss_id]
            with st.expander(f'📨 {req.request_id}  |  {req.mh_id} → MSS_{req.source_mss_id}  |  P={req.priority}  |  {req.status}'):
                st.write(f'**Origin:** {req.mh_id} at **MSS_{req.source_mss_id}**')
                st.write(f'**Priority:** {req.priority}  |  **Lamport Time:** {req.timestamp}')
                st.write(f'**Broadcast to:** {", ".join(other_mss)}')
                st.write(f'**Messages generated:** {len(other_mss)} (O(N−1) = {ring.n - 1})')
                st.write(f'**Current status:** `{req.status}`')
    else:
        st.warning('No requests yet. Use the sidebar to send requests.')
    
    st.markdown('---')
    st.markdown('#### Global Priority Queue (Token Holder\'s View)')
    holder = ring.token_holder()
    if holder and holder.global_queue:
        gq = []
        for rank, r in enumerate(holder.global_queue, 1):
            is_local = '✅' if r.source_mss_id == holder.id else ''
            gq.append({
                'Rank': rank,
                'MH': r.mh_id,
                'Source MSS': f'MSS_{r.source_mss_id}',
                'Priority': r.priority,
                'Lamport T': r.timestamp,
                'Local?': is_local,
                'Status': r.status,
            })
        st.dataframe(pd.DataFrame(gq), use_container_width=True, hide_index=True)
        st.caption(f'Token at **MSS_{holder.id}** — only **local** PENDING requests can be granted.')
    else:
        st.info('Global queue is empty or no token holder.')
    
    st.markdown('---')
    st.markdown('#### Event Log')
    if tm.event_log:
        for ev in tm.event_log[-15:]:
            st.text(ev)
    else:
        st.info('No events yet. Step the simulation.')


# ────────── TAB 3: REPLICATED LOGS & PRIORITIES ──────────
with tab3:
    st.markdown('### 3 — Replicated Request Logs at Each MSS')
    
    st.info('''
    Every MSS keeps a **complete replicated copy** of all requests.
    This ensures any MSS can determine the correct priority order.
    ''')
    
    for mss in ring.nodes:
        token_badge = ' 🔑 (Token Holder)' if mss.has_token else ''
        with st.expander(f'📋 MSS_{mss.id}{token_badge}  —  {len(mss.replicated_log)} log entries', expanded=True):
            if mss.replicated_log:
                log_data = []
                for idx_r, r in enumerate(mss.replicated_log, 1):
                    is_local = 'Local' if r.source_mss_id == mss.id else 'Replicated'
                    log_data.append({
                        '#': idx_r,
                        'Request ID': r.request_id,
                        'MH': r.mh_id,
                        'Origin MSS': f'MSS_{r.source_mss_id}',
                        'Priority': r.priority,
                        'Lamport T': r.timestamp,
                        'Type': is_local,
                        'Status': r.status,
                    })
                df = pd.DataFrame(log_data)
                st.dataframe(df, use_container_width=True, hide_index=True)
            else:
                st.caption('No entries yet')
    
    st.markdown('---')
    st.markdown('#### Verification: All MSSs Have Identical Request Sets')
    sets = []
    for mss in ring.nodes:
        ids = sorted(set(r.request_id for r in mss.replicated_log))
        sets.append(ids)
    if len(sets) > 1 and all(s == sets[0] for s in sets):
        st.success('✅ All MSSs have identical replicated request sets (replication is consistent)')
    elif not any(sets):
        st.info('No requests to compare yet')
    else:
        st.warning('⚠️ Logs not yet synchronized (requests may still be propagating)')


# ────────── TAB 4: QUEUES AFTER SERVICE ──────────
with tab4:
    st.markdown('### 4 — Queue State After Requests Have Been Served')
    
    st.info('''
    After a request is **granted** and later **completed**, it is removed from the
    global queue. The tables below show the **current** queue state at every MSS.
    ''')
    
    cols = st.columns(ring.n)
    for idx_col, mss in enumerate(ring.nodes):
        with cols[idx_col]:
            tok = ' 🔑' if mss.has_token else ''
            st.markdown(f'#### MSS_{mss.id}{tok}')
            
            st.caption('**Local Queue (own MHs)**')
            if mss.local_queue:
                ldf = pd.DataFrame([{
                    'MH': r.mh_id,
                    'P': r.priority,
                    'T': r.timestamp,
                    'Status': r.status,
                } for r in mss.local_queue])
                st.dataframe(ldf, hide_index=True, use_container_width=True)
            else:
                st.success('Empty ✓')
            
            st.caption('**Global Priority Queue**')
            if mss.global_queue:
                gdf = pd.DataFrame([{
                    'Rank': i + 1,
                    'MH': r.mh_id,
                    'MSS': r.source_mss_id,
                    'P': r.priority,
                    'T': r.timestamp,
                    'Status': r.status,
                } for i, r in enumerate(mss.global_queue)])
                st.dataframe(gdf, hide_index=True, use_container_width=True)
            else:
                st.success('Empty ✓')
    
    st.markdown('---')
    st.markdown('#### Completed Requests (Served & Removed from Queues)')
    completed = []
    seen_ids = set()
    for mss in ring.nodes:
        for r in mss.replicated_log:
            if r.status == 'COMPLETED' and r.request_id not in seen_ids:
                completed.append(r.to_dict())
                seen_ids.add(r.request_id)
    if completed:
        st.dataframe(pd.DataFrame(completed), use_container_width=True, hide_index=True)
    else:
        st.warning('No completed requests yet. Grant and complete some requests using the sidebar.')
    
    st.markdown('---')
    st.markdown('#### Per-MSS Statistics')
    st.dataframe(
        pd.DataFrame([mss.stats() for mss in ring.nodes]),
        use_container_width=True,
        hide_index=True,
    )
    
    total_msgs = sum(m.messages_sent + m.messages_received for m in ring.nodes)
    total_reqs = len(set(r.request_id for r in ring.nodes[0].replicated_log)) if ring.nodes[0].replicated_log else 0
    avg_str = f'{total_msgs / total_reqs:.1f}' if total_reqs else '—'
    
    st.markdown('#### Message Complexity')
    st.info(f'''
    **N (MSSs):** {ring.n}  
    **Messages per broadcast:** N − 1 = {ring.n - 1}  
    **Total unique requests:** {total_reqs}  
    **Total messages exchanged:** {total_msgs}  
    **Avg messages / request:** {avg_str}
    ''')


# Footer
st.markdown('---')
st.markdown('''
<div style='text-align:center;color:gray;font-size:.85rem;'>
<b>Token-Ring Mutual Exclusion with Replication</b> — 
MSS-MH Architecture · Lamport Clocks · Priority Queues · Request Broadcasting
</div>
''', unsafe_allow_html=True)
