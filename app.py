"""
Token-Ring Mutual Exclusion with Replication — Fully Automated Random Scenarios
================================================================================
All tabs auto-generate random requests with random priorities.
User only clicks "Generate" then "Play" — everything animates automatically.
"""

import streamlit as st
import pandas as pd
import numpy as np
import math
import random
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple
import plotly.graph_objects as go

# ═══════════════════════════════════════════════════════════════
#                        MODEL CLASSES
# ═══════════════════════════════════════════════════════════════

@dataclass
class Request:
    mh_id: str
    source_mss: int
    priority: int
    timestamp: int
    status: str = "PENDING"
    rid: str = field(default="", init=False)
    
    def __post_init__(self):
        self.rid = f"REQ_{self.mh_id}_T{self.timestamp}"
    
    def row(self):
        return {
            "Request ID": self.rid,
            "Mobile Host": self.mh_id,
            "Source MSS": f"MSS_{self.source_mss}",
            "Priority": self.priority,
            "Timestamp": self.timestamp,
            "Status": self.status
        }


class Geom:
    """Shared geometry for all visualizations."""
    def __init__(self, n, r=2.2, mr=0.55):
        self.n = n
        self.R = r
        self.mr = mr
        self.ang = [2 * math.pi * i / n - math.pi / 2 for i in range(n)]
        self.sx = [r * math.cos(a) for a in self.ang]
        self.sy = [r * math.sin(a) for a in self.ang]
        self.hp, self.hn = {}, {}
        for s in range(n):
            self.hp[s], self.hn[s] = [], []
            for j in range(3):
                a2 = self.ang[s] + math.pi + (j - 1) * 0.4
                self.hp[s].append((self.sx[s] + mr * math.cos(a2),
                                   self.sy[s] + mr * math.sin(a2)))
                self.hn[s].append(f"MH_{s}_{chr(65 + j)}")
    
    def all_mhs(self):
        """Return list of all (mss_id, mh_idx, mh_name) tuples."""
        result = []
        for s in range(self.n):
            for j in range(3):
                result.append((s, j, self.hn[s][j]))
        return result


def lerp(a, b, t):
    return a + (b - a) * max(0.0, min(1.0, t))


def generate_random_scenario(g: Geom, min_requests=3, max_requests=8):
    """
    Generate a random scenario:
    - Pick random number of MHs (between min_requests and max_requests)
    - Assign random priorities (1-10)
    - Shuffle to determine request order (which sets Lamport timestamps)
    
    Returns list of Request objects.
    """
    all_mhs = g.all_mhs()
    num_requests = random.randint(min_requests, min(max_requests, len(all_mhs)))
    selected = random.sample(all_mhs, num_requests)
    random.shuffle(selected)  # Shuffle determines arrival order
    
    requests = []
    for timestamp, (mss_id, mh_idx, mh_name) in enumerate(selected, start=1):
        priority = random.randint(1, 10)
        req = Request(
            mh_id=mh_name,
            source_mss=mss_id,
            priority=priority,
            timestamp=timestamp
        )
        requests.append(req)
    
    return requests


# ═══════════════════════════════════════════════════════════════
#                    BROADCASTING ANIMATION
# ═══════════════════════════════════════════════════════════════

class BroadcastAnimator:
    """
    Animates multiple random requests being broadcast.
    Each request: MH → local MSS → all other MSSs
    """
    
    def __init__(self, g: Geom, requests: List[Request]):
        self.g = g
        self.requests = requests
        self.frames = []
        self.logs = []
        
        # Timing
        self.MSG_STEPS = 6
        self.HOLD_STEPS = 4
        
        # Track queue counts at each MSS
        self.queue_counts: Dict[int, int] = {i: 0 for i in range(g.n)}
        # Track which MSSs have received each request
        self.mss_has_request: Dict[int, set] = {i: set() for i in range(g.n)}
    
    def _build_frame(self, highlight_mh=None, highlight_mss=None, 
                     msg_pos=None, msg_label="REQ", arrows=None,
                     log_text="", all_green=False):
        """Build a single frame with consistent trace count."""
        g = self.g
        data = []
        
        # 0: Ring
        ra = np.linspace(0, 2 * np.pi, 100)
        data.append(go.Scatter(
            x=g.R * 1.08 * np.cos(ra), y=g.R * 1.08 * np.sin(ra),
            mode='lines', line=dict(color='#888', width=2),
            hoverinfo='none', showlegend=False
        ))
        
        # 1: MSS nodes
        mss_colors = []
        mss_sizes = []
        for i in range(g.n):
            if all_green:
                mss_colors.append('#00CC00')
                mss_sizes.append(55)
            elif highlight_mss and i in highlight_mss:
                mss_colors.append(highlight_mss[i])
                mss_sizes.append(55)
            else:
                mss_colors.append('#00D4FF')
                mss_sizes.append(48)
        
        data.append(go.Scatter(
            x=g.sx, y=g.sy, mode='markers+text',
            marker=dict(size=mss_sizes, color=mss_colors,
                        line=dict(width=3, color='white'), symbol='square'),
            text=[f'MSS_{i}' for i in range(g.n)],
            textposition='top center',
            textfont=dict(size=11, color='#333', family='Arial Black'),
            hoverinfo='none', showlegend=False
        ))
        
        # 2: MH nodes
        mh_x, mh_y, mh_colors, mh_sizes, mh_text = [], [], [], [], []
        for s in range(g.n):
            for j in range(3):
                mx, my = g.hp[s][j]
                mh_name = g.hn[s][j]
                mh_x.append(mx)
                mh_y.append(my)
                mh_text.append(mh_name.split('_')[-1])
                
                if highlight_mh and mh_name == highlight_mh:
                    mh_colors.append('#FF5722')
                    mh_sizes.append(30)
                else:
                    mh_colors.append('#4CAF50')
                    mh_sizes.append(20)
        
        data.append(go.Scatter(
            x=mh_x, y=mh_y, mode='markers+text',
            marker=dict(size=mh_sizes, color=mh_colors,
                        line=dict(width=2, color='white')),
            text=mh_text, textposition='bottom center',
            textfont=dict(size=9, color='#333'),
            hoverinfo='none', showlegend=False
        ))
        
        # 3: Connection lines
        conn_x, conn_y = [], []
        for s in range(g.n):
            for j in range(3):
                mx, my = g.hp[s][j]
                conn_x.extend([g.sx[s], mx, None])
                conn_y.extend([g.sy[s], my, None])
        
        data.append(go.Scatter(
            x=conn_x, y=conn_y, mode='lines',
            line=dict(color='#ccc', width=1, dash='dot'),
            hoverinfo='none', showlegend=False
        ))
        
        # 4: Broadcast arrows (persistent)
        if arrows:
            data.append(go.Scatter(
                x=arrows['x'], y=arrows['y'], mode='lines',
                line=dict(color='#FF5722', width=2),
                hoverinfo='none', showlegend=False
            ))
        else:
            data.append(go.Scatter(x=[None], y=[None], mode='lines',
                                   hoverinfo='none', showlegend=False))
        
        # 5: Traveling message
        if msg_pos:
            data.append(go.Scatter(
                x=[msg_pos[0]], y=[msg_pos[1]], mode='markers+text',
                marker=dict(size=20, color='#FF5722', symbol='diamond',
                            line=dict(width=2, color='white')),
                text=[msg_label], textposition='top center',
                textfont=dict(size=10, color='#FF5722', family='Arial Black'),
                hoverinfo='none', showlegend=False
            ))
        else:
            data.append(go.Scatter(x=[None], y=[None], mode='markers',
                                   opacity=0, hoverinfo='none', showlegend=False))
        
        # 6: Queue count badges
        badge_x, badge_y, badge_text = [], [], []
        for i in range(g.n):
            badge_x.append(g.sx[i])
            badge_y.append(g.sy[i] - 0.4)
            cnt = self.queue_counts.get(i, 0)
            badge_text.append(f"Q:{cnt}" if cnt > 0 else "")
        
        data.append(go.Scatter(
            x=badge_x, y=badge_y, mode='text',
            text=badge_text,
            textfont=dict(size=12, color='#FF5722', family='Arial Black'),
            hoverinfo='none', showlegend=False
        ))
        
        # 7: Log text
        data.append(go.Scatter(
            x=[0], y=[-3.1], mode='text',
            text=[f'<b>{log_text}</b>'],
            textfont=dict(size=14, color='#333'),
            hoverinfo='none', showlegend=False
        ))
        
        return data
    
    def build(self):
        """Build the complete broadcast animation for all requests."""
        g = self.g
        arrows_x, arrows_y = [], []
        
        for req_idx, req in enumerate(self.requests):
            src_mss = req.source_mss
            mh_name = req.mh_id
            mh_idx = ord(mh_name.split('_')[2]) - 65
            mh_x, mh_y = g.hp[src_mss][mh_idx]
            mss_x, mss_y = g.sx[src_mss], g.sy[src_mss]
            
            self.logs.append(f"Request {req_idx + 1}: {mh_name} (Priority={req.priority}) → MSS_{src_mss}")
            
            # Phase A: MH sends to local MSS
            for step in range(self.MSG_STEPS):
                t = (step + 1) / self.MSG_STEPS
                mx = lerp(mh_x, mss_x, t)
                my = lerp(mh_y, mss_y, t)
                
                self.frames.append(self._build_frame(
                    highlight_mh=mh_name,
                    highlight_mss={src_mss: '#FFD700'},
                    msg_pos=(mx, my),
                    msg_label=f"P{req.priority}",
                    arrows={'x': list(arrows_x), 'y': list(arrows_y)} if arrows_x else None,
                    log_text=f"📤 {mh_name} (Priority {req.priority}) → MSS_{src_mss}"
                ))
            
            # Update queue at source MSS
            self.queue_counts[src_mss] += 1
            self.mss_has_request[src_mss].add(req.rid)
            
            # Phase B: Hold at source MSS
            for _ in range(self.HOLD_STEPS):
                self.frames.append(self._build_frame(
                    highlight_mh=mh_name,
                    highlight_mss={src_mss: '#00CC00'},
                    arrows={'x': list(arrows_x), 'y': list(arrows_y)} if arrows_x else None,
                    log_text=f"📋 MSS_{src_mss} queued request (P={req.priority}, T={req.timestamp})"
                ))
            
            # Phase C: Broadcast to all other MSSs
            for tgt_mss in range(g.n):
                if tgt_mss == src_mss:
                    continue
                
                tgt_x, tgt_y = g.sx[tgt_mss], g.sy[tgt_mss]
                
                # Animate message traveling
                for step in range(self.MSG_STEPS):
                    t = (step + 1) / self.MSG_STEPS
                    mx = lerp(mss_x, tgt_x, t)
                    my = lerp(mss_y, tgt_y, t)
                    
                    recv_mss = {i: '#87CEEB' for i in range(g.n) 
                               if i != src_mss and req.rid in self.mss_has_request[i]}
                    recv_mss[src_mss] = '#00CC00'
                    
                    self.frames.append(self._build_frame(
                        highlight_mh=mh_name,
                        highlight_mss=recv_mss,
                        msg_pos=(mx, my),
                        msg_label=f"P{req.priority}",
                        arrows={'x': list(arrows_x), 'y': list(arrows_y)} if arrows_x else None,
                        log_text=f"📡 Broadcasting to MSS_{tgt_mss}..."
                    ))
                
                # Add arrow
                arrows_x.extend([mss_x, tgt_x, None])
                arrows_y.extend([mss_y, tgt_y, None])
                
                # Target MSS receives
                self.queue_counts[tgt_mss] += 1
                self.mss_has_request[tgt_mss].add(req.rid)
                
                recv_mss = {i: '#87CEEB' for i in range(g.n) 
                           if i != src_mss and req.rid in self.mss_has_request[i]}
                recv_mss[src_mss] = '#00CC00'
                recv_mss[tgt_mss] = '#00CC00'
                
                for _ in range(self.HOLD_STEPS):
                    self.frames.append(self._build_frame(
                        highlight_mh=mh_name,
                        highlight_mss=recv_mss,
                        arrows={'x': list(arrows_x), 'y': list(arrows_y)},
                        log_text=f"✅ MSS_{tgt_mss} received & replicated!"
                    ))
            
            self.logs.append(f"  → Broadcast complete: all {g.n} MSSs have this request")
        
        # Final frames: all green
        for _ in range(self.HOLD_STEPS * 3):
            self.frames.append(self._build_frame(
                all_green=True,
                arrows={'x': arrows_x, 'y': arrows_y},
                log_text=f"✅ All {len(self.requests)} requests replicated to all {g.n} MSSs!"
            ))
        
        # Package as Plotly animation
        plotly_frames = [go.Frame(data=f, name=str(i)) for i, f in enumerate(self.frames)]
        
        layout = go.Layout(
            title=dict(text='<b>Request Broadcasting: Random MHs → All MSSs</b>',
                       font=dict(size=16), x=0.5),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-3.3, 3.3]),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, 
                       range=[-3.5, 3.3], scaleanchor='x'),
            height=650, plot_bgcolor='#fafafa', paper_bgcolor='#fafafa',
            margin=dict(l=20, r=20, t=60, b=20), showlegend=False,
            updatemenus=[dict(
                type='buttons', showactive=False, x=0.05, y=-0.02,
                xanchor='left', yanchor='top', direction='right',
                buttons=[
                    dict(label='▶ Play', method='animate',
                         args=[None, dict(frame=dict(duration=40, redraw=True),
                                         fromcurrent=True, transition=dict(duration=0))]),
                    dict(label='⏸ Pause', method='animate',
                         args=[[None], dict(frame=dict(duration=0, redraw=False),
                                           mode='immediate')])
                ]
            )],
            sliders=[dict(
                active=0, yanchor='top', xanchor='left',
                currentvalue=dict(font=dict(size=11), prefix='Step: ', visible=True),
                transition=dict(duration=0), pad=dict(b=10, t=30),
                len=0.75, x=0.2, y=-0.02,
                steps=[dict(args=[[str(i)], dict(frame=dict(duration=0, redraw=True),
                            mode='immediate')], label='', method='animate')
                       for i in range(len(plotly_frames))]
            )]
        )
        
        return self.frames[0], plotly_frames, layout, self.logs


# ═══════════════════════════════════════════════════════════════
#                 PRIORITY GRANTING ANIMATION
# ═══════════════════════════════════════════════════════════════

class GrantingAnimator:
    """
    Animates token circulation and priority-based granting.
    Token visits each MSS; if pending local requests exist, grants to highest priority.
    """
    
    def __init__(self, g: Geom, requests: List[Request]):
        self.g = g
        self.requests = [Request(r.mh_id, r.source_mss, r.priority, r.timestamp) 
                        for r in requests]  # Deep copy
        self.frames = []
        self.logs = []
        self.token_pos = 0.0
        
        # Timing
        self.MOVE_STEPS = 10
        self.HOLD_STEPS = 8
        self.GRANT_STEPS = 12
    
    def _get_pending_at_mss(self, mss_id):
        """Get pending requests at a specific MSS, sorted by priority."""
        pending = [r for r in self.requests 
                   if r.source_mss == mss_id and r.status == "PENDING"]
        return sorted(pending, key=lambda r: (-r.priority, r.timestamp))
    
    def _build_frame(self, token_color='#FFF', mss_colors=None, mh_colors=None,
                     bar_data=None, granting_mh=None, log_text=""):
        """Build frame with token, ring, and priority bar chart."""
        g = self.g
        data = []
        mss_colors = mss_colors or {}
        mh_colors = mh_colors or {}
        
        # 0: Ring
        ra = np.linspace(0, 2 * np.pi, 100)
        data.append(go.Scatter(
            x=g.R * 1.08 * np.cos(ra), y=g.R * 1.08 * np.sin(ra),
            mode='lines', line=dict(color='#555', width=2),
            hoverinfo='none', showlegend=False
        ))
        
        # 1: MSS nodes
        mss_cols = [mss_colors.get(i, '#00D4FF') for i in range(g.n)]
        mss_sizes = [55 if i in mss_colors else 48 for i in range(g.n)]
        
        data.append(go.Scatter(
            x=g.sx, y=g.sy, mode='markers+text',
            marker=dict(size=mss_sizes, color=mss_cols,
                        line=dict(width=3, color='white'), symbol='square'),
            text=[f'MSS_{i}' for i in range(g.n)],
            textposition='top center',
            textfont=dict(size=11, color='white', family='Arial Black'),
            hoverinfo='none', showlegend=False
        ))
        
        # 2: MH nodes
        mh_x, mh_y, mh_cols, mh_sizes = [], [], [], []
        for s in range(g.n):
            for j in range(3):
                mx, my = g.hp[s][j]
                mh_name = g.hn[s][j]
                mh_x.append(mx)
                mh_y.append(my)
                mh_cols.append(mh_colors.get(mh_name, '#4CAF50'))
                mh_sizes.append(28 if mh_name in mh_colors else 20)
        
        data.append(go.Scatter(
            x=mh_x, y=mh_y, mode='markers',
            marker=dict(size=mh_sizes, color=mh_cols,
                        line=dict(width=2, color='white')),
            hoverinfo='none', showlegend=False
        ))
        
        # 3: Token
        token_angle = ((self.token_pos % g.n) / g.n) * 2 * math.pi - math.pi / 2
        tx = g.R * math.cos(token_angle)
        ty = g.R * math.sin(token_angle)
        
        data.append(go.Scatter(
            x=[tx], y=[ty], mode='markers+text',
            marker=dict(size=38, color=token_color, symbol='circle',
                        line=dict(width=4, color='#333')),
            text=['🔑'], textfont=dict(size=16),
            hoverinfo='none', showlegend=False
        ))
        
        # 4: Priority bars (mini bar chart on the right side)
        if bar_data:
            bar_y = 2.5
            bar_x_start = 3.8
            bar_height = 0.25
            bar_spacing = 0.35
            
            for i, (mh, pri, status, is_next) in enumerate(bar_data[:8]):  # Max 8 shown
                bar_width = pri * 0.15
                
                if is_next:
                    color = '#00CC00'
                elif status == 'GRANTED':
                    color = '#9C27B0'
                elif status == 'COMPLETED':
                    color = '#888'
                else:
                    color = '#FF5722'
                
                y_pos = bar_y - i * bar_spacing
                
                # Bar rectangle (approximated with thick line)
                data.append(go.Scatter(
                    x=[bar_x_start, bar_x_start + bar_width],
                    y=[y_pos, y_pos],
                    mode='lines',
                    line=dict(width=18, color=color),
                    hoverinfo='none', showlegend=False
                ))
                
                # Label
                data.append(go.Scatter(
                    x=[bar_x_start - 0.1], y=[y_pos],
                    mode='text',
                    text=[f"{mh.split('_')[-1]} P{pri}"],
                    textposition='middle left',
                    textfont=dict(size=9, color='white'),
                    hoverinfo='none', showlegend=False
                ))
        else:
            # Placeholder traces to maintain count
            for _ in range(16):
                data.append(go.Scatter(x=[None], y=[None], mode='lines',
                                       hoverinfo='none', showlegend=False))
        
        # 5: Log text
        data.append(go.Scatter(
            x=[0], y=[-3.2], mode='text',
            text=[f'<b>{log_text}</b>'],
            textfont=dict(size=13, color='white'),
            hoverinfo='none', showlegend=False
        ))
        
        return data
    
    def _get_bar_data(self, current_mss):
        """Get bar chart data sorted by priority."""
        all_reqs = sorted(self.requests, key=lambda r: (-r.priority, r.timestamp))
        
        # Find what would be granted next at current MSS
        local_pending = self._get_pending_at_mss(current_mss)
        next_grant = local_pending[0] if local_pending else None
        
        bar_data = []
        for r in all_reqs:
            is_next = (r is next_grant)
            bar_data.append((r.mh_id, r.priority, r.status, is_next))
        
        return bar_data
    
    def build(self):
        """Build the complete granting animation."""
        g = self.g
        
        # Determine which MSSs have pending requests
        mss_with_pending = set(r.source_mss for r in self.requests if r.status == "PENDING")
        max_mss = max(mss_with_pending) if mss_with_pending else g.n - 1
        
        # Token starts at MSS_0 and visits each MSS
        self.logs.append("Token begins circulation from MSS_0")
        
        granted_count = 0
        
        for target_mss in range(max_mss + 2):  # Visit enough MSSs
            actual_mss = target_mss % g.n
            
            # Move token to this MSS
            start_pos = self.token_pos
            end_pos = target_mss
            
            for step in range(self.MOVE_STEPS):
                t = (step + 1) / self.MOVE_STEPS
                self.token_pos = lerp(start_pos, end_pos, t)
                
                bar_data = self._get_bar_data(actual_mss)
                
                self.frames.append(self._build_frame(
                    token_color='#FFF',
                    mss_colors={actual_mss: '#87CEEB'} if step > self.MOVE_STEPS // 2 else {},
                    bar_data=bar_data,
                    log_text=f"⚪ Token moving to MSS_{actual_mss}..."
                ))
            
            # Check for pending requests at this MSS
            local_pending = self._get_pending_at_mss(actual_mss)
            
            if local_pending:
                grantee = local_pending[0]
                
                self.logs.append(f"MSS_{actual_mss}: Granting to {grantee.mh_id} "
                                f"(Priority={grantee.priority}, T={grantee.timestamp})")
                
                # Hold token (checking)
                bar_data = self._get_bar_data(actual_mss)
                for _ in range(self.HOLD_STEPS):
                    self.frames.append(self._build_frame(
                        token_color='#FFD700',
                        mss_colors={actual_mss: '#FFD700'},
                        mh_colors={grantee.mh_id: '#FF5722'},
                        bar_data=bar_data,
                        log_text=f"🔍 MSS_{actual_mss}: Found {len(local_pending)} pending request(s)"
                    ))
                
                # Grant animation
                for _ in range(self.GRANT_STEPS):
                    self.frames.append(self._build_frame(
                        token_color='#00FF00',
                        mss_colors={actual_mss: '#00FF00'},
                        mh_colors={grantee.mh_id: '#9C27B0'},
                        bar_data=bar_data,
                        log_text=f"🏆 GRANTED to {grantee.mh_id} (Priority {grantee.priority} wins!)"
                    ))
                
                # Update status
                grantee.status = "GRANTED"
                granted_count += 1
                
                # Critical section
                bar_data = self._get_bar_data(actual_mss)
                for _ in range(self.GRANT_STEPS):
                    self.frames.append(self._build_frame(
                        token_color='#00FF00',
                        mss_colors={actual_mss: '#00FF00'},
                        mh_colors={grantee.mh_id: '#9C27B0'},
                        bar_data=bar_data,
                        log_text=f"🟣 {grantee.mh_id} in CRITICAL SECTION"
                    ))
                
                # Complete
                grantee.status = "COMPLETED"
                bar_data = self._get_bar_data(actual_mss)
                for _ in range(self.HOLD_STEPS):
                    self.frames.append(self._build_frame(
                        token_color='#FFF',
                        mss_colors={actual_mss: '#00CC00'},
                        bar_data=bar_data,
                        log_text=f"✅ {grantee.mh_id} completed, releasing token"
                    ))
            
            else:
                # No pending requests, brief check
                bar_data = self._get_bar_data(actual_mss)
                for _ in range(self.HOLD_STEPS):
                    self.frames.append(self._build_frame(
                        token_color='#FFF',
                        mss_colors={actual_mss: '#87CEEB'},
                        bar_data=bar_data,
                        log_text=f"🔍 MSS_{actual_mss}: No local pending → passing token"
                    ))
                
                self.logs.append(f"MSS_{actual_mss}: No local pending requests, token passes")
            
            # Check if all done
            remaining = sum(1 for r in self.requests if r.status == "PENDING")
            if remaining == 0:
                break
        
        # Final frames
        for _ in range(self.HOLD_STEPS * 2):
            self.frames.append(self._build_frame(
                token_color='#FFF',
                bar_data=self._get_bar_data(0),
                log_text=f"✅ All {len(self.requests)} requests have been served!"
            ))
        
        self.logs.append(f"Complete: {granted_count} requests granted based on priority order")
        
        # Package
        plotly_frames = [go.Frame(data=f, name=str(i)) for i, f in enumerate(self.frames)]
        
        layout = go.Layout(
            title=dict(text='<b>Priority-Based Token Granting</b>', font=dict(size=16), x=0.5),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-3.5, 6]),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False,
                       range=[-3.6, 3.3], scaleanchor='x'),
            height=650, plot_bgcolor='#1a1a2e', paper_bgcolor='#1a1a2e',
            margin=dict(l=20, r=20, t=60, b=20), showlegend=False,
            updatemenus=[dict(
                type='buttons', showactive=False, x=0.05, y=-0.02,
                xanchor='left', yanchor='top', direction='right',
                buttons=[
                    dict(label='▶ Play', method='animate',
                         args=[None, dict(frame=dict(duration=50, redraw=True),
                                         fromcurrent=True, transition=dict(duration=0))]),
                    dict(label='⏸ Pause', method='animate',
                         args=[[None], dict(frame=dict(duration=0, redraw=False),
                                           mode='immediate')])
                ]
            )],
            sliders=[dict(
                active=0, yanchor='top', xanchor='left',
                currentvalue=dict(font=dict(size=11, color='white'), prefix='Step: ', visible=True),
                transition=dict(duration=0), pad=dict(b=10, t=30),
                len=0.7, x=0.2, y=-0.02,
                steps=[dict(args=[[str(i)], dict(frame=dict(duration=0, redraw=True),
                            mode='immediate')], label='', method='animate')
                       for i in range(len(plotly_frames))]
            )]
        )
        
        return self.frames[0], plotly_frames, layout, self.logs, self.requests


# ═══════════════════════════════════════════════════════════════
#                    QUEUE STATE ANIMATION
# ═══════════════════════════════════════════════════════════════

class QueueAnimator:
    """
    Animates the queue states at each MSS as requests are processed.
    Shows pending → granted → completed transitions.
    """
    
    def __init__(self, g: Geom, requests: List[Request]):
        self.g = g
        self.requests = [Request(r.mh_id, r.source_mss, r.priority, r.timestamp)
                        for r in requests]
        self.frames = []
        self.logs = []
        self.token_pos = 0.0
        
        self.MOVE_STEPS = 8
        self.PROCESS_STEPS = 15
    
    def _build_frame(self, token_color='#FFF', mss_colors=None, queue_states=None, log_text=""):
        """Build frame showing queue states at each MSS."""
        g = self.g
        data = []
        mss_colors = mss_colors or {}
        queue_states = queue_states or {}
        
        # 0: Ring (smaller, to make room for queues)
        ra = np.linspace(0, 2 * np.pi, 100)
        r_small = g.R * 0.85
        data.append(go.Scatter(
            x=r_small * 1.08 * np.cos(ra), y=r_small * 1.08 * np.sin(ra),
            mode='lines', line=dict(color='#555', width=2),
            hoverinfo='none', showlegend=False
        ))
        
        # 1: MSS nodes (smaller positions)
        sx = [r_small * math.cos(a) for a in g.ang]
        sy = [r_small * math.sin(a) for a in g.ang]
        
        mss_cols = [mss_colors.get(i, '#00D4FF') for i in range(g.n)]
        
        data.append(go.Scatter(
            x=sx, y=sy, mode='markers+text',
            marker=dict(size=45, color=mss_cols,
                        line=dict(width=2, color='white'), symbol='square'),
            text=[f'MSS_{i}' for i in range(g.n)],
            textposition='middle center',
            textfont=dict(size=10, color='white', family='Arial Black'),
            hoverinfo='none', showlegend=False
        ))
        
        # 2: Token
        token_angle = ((self.token_pos % g.n) / g.n) * 2 * math.pi - math.pi / 2
        tx = r_small * math.cos(token_angle)
        ty = r_small * math.sin(token_angle)
        
        data.append(go.Scatter(
            x=[tx], y=[ty], mode='markers+text',
            marker=dict(size=30, color=token_color, symbol='circle',
                        line=dict(width=3, color='#333')),
            text=['🔑'], textfont=dict(size=12),
            hoverinfo='none', showlegend=False
        ))
        
        # 3-8: Queue displays for each MSS (positioned around the ring)
        for mss_id in range(min(g.n, 6)):  # Max 6 MSS displays
            angle = g.ang[mss_id]
            qx = g.R * 1.6 * math.cos(angle)
            qy = g.R * 1.6 * math.sin(angle)
            
            qs = queue_states.get(mss_id, {'pending': [], 'granted': [], 'completed': []})
            
            # Queue box text
            p_count = len(qs['pending'])
            g_count = len(qs['granted'])
            c_count = len(qs['completed'])
            
            lines = [f"<b>MSS_{mss_id}</b>"]
            if p_count > 0:
                lines.append(f"⏳ Pending: {p_count}")
                for r in qs['pending'][:2]:
                    lines.append(f"  • {r.mh_id.split('_')[-1]} P{r.priority}")
            if g_count > 0:
                lines.append(f"🟢 Granted: {g_count}")
            if c_count > 0:
                lines.append(f"✅ Done: {c_count}")
            if p_count == 0 and g_count == 0 and c_count == 0:
                lines.append("(empty)")
            
            text = "<br>".join(lines)
            
            # Background color based on state
            if g_count > 0:
                bg_color = 'rgba(0, 200, 0, 0.2)'
            elif p_count > 0:
                bg_color = 'rgba(255, 200, 0, 0.2)'
            else:
                bg_color = 'rgba(100, 100, 100, 0.1)'
            
            data.append(go.Scatter(
                x=[qx], y=[qy], mode='text',
                text=[text],
                textfont=dict(size=10, color='white'),
                hoverinfo='none', showlegend=False
            ))
        
        # Pad to consistent trace count
        while len(data) < 10:
            data.append(go.Scatter(x=[None], y=[None], mode='markers',
                                   opacity=0, hoverinfo='none', showlegend=False))
        
        # Log text
        data.append(go.Scatter(
            x=[0], y=[-3.0], mode='text',
            text=[f'<b>{log_text}</b>'],
            textfont=dict(size=13, color='white'),
            hoverinfo='none', showlegend=False
        ))
        
        return data
    
    def _get_queue_states(self):
        """Get current queue state at each MSS."""
        states = {}
        for mss_id in range(self.g.n):
            states[mss_id] = {
                'pending': [r for r in self.requests 
                           if r.source_mss == mss_id and r.status == "PENDING"],
                'granted': [r for r in self.requests 
                           if r.source_mss == mss_id and r.status == "GRANTED"],
                'completed': [r for r in self.requests 
                             if r.source_mss == mss_id and r.status == "COMPLETED"]
            }
            # Sort pending by priority
            states[mss_id]['pending'].sort(key=lambda r: (-r.priority, r.timestamp))
        return states
    
    def build(self):
        """Build queue state animation."""
        g = self.g
        
        # Initial state
        self.logs.append("Initial state: all requests pending")
        qs = self._get_queue_states()
        for _ in range(self.PROCESS_STEPS):
            self.frames.append(self._build_frame(
                queue_states=qs,
                log_text=f"📋 Initial: {len(self.requests)} requests pending across {g.n} MSSs"
            ))
        
        # Process requests in token order
        mss_with_pending = set(r.source_mss for r in self.requests)
        max_mss = max(mss_with_pending) if mss_with_pending else g.n - 1
        
        for target_mss in range(max_mss + 2):
            actual_mss = target_mss % g.n
            
            # Move token
            start_pos = self.token_pos
            for step in range(self.MOVE_STEPS):
                t = (step + 1) / self.MOVE_STEPS
                self.token_pos = lerp(start_pos, target_mss, t)
                qs = self._get_queue_states()
                self.frames.append(self._build_frame(
                    mss_colors={actual_mss: '#87CEEB'},
                    queue_states=qs,
                    log_text=f"⚪ Token → MSS_{actual_mss}"
                ))
            
            # Check queue
            qs = self._get_queue_states()
            pending = qs[actual_mss]['pending']
            
            if pending:
                grantee = pending[0]
                self.logs.append(f"MSS_{actual_mss}: Serving {grantee.mh_id} (P={grantee.priority})")
                
                # Grant
                grantee.status = "GRANTED"
                qs = self._get_queue_states()
                for _ in range(self.PROCESS_STEPS):
                    self.frames.append(self._build_frame(
                        token_color='#00FF00',
                        mss_colors={actual_mss: '#00FF00'},
                        queue_states=qs,
                        log_text=f"🟢 MSS_{actual_mss}: Granted to {grantee.mh_id}"
                    ))
                
                # Complete
                grantee.status = "COMPLETED"
                qs = self._get_queue_states()
                for _ in range(self.PROCESS_STEPS):
                    self.frames.append(self._build_frame(
                        token_color='#FFF',
                        mss_colors={actual_mss: '#888'},
                        queue_states=qs,
                        log_text=f"✅ MSS_{actual_mss}: {grantee.mh_id} completed"
                    ))
            else:
                for _ in range(self.MOVE_STEPS):
                    self.frames.append(self._build_frame(
                        mss_colors={actual_mss: '#555'},
                        queue_states=qs,
                        log_text=f"MSS_{actual_mss}: Queue empty, passing"
                    ))
            
            # Check if done
            remaining = sum(1 for r in self.requests if r.status == "PENDING")
            if remaining == 0:
                break
        
        # Final state
        self.logs.append("Final state: all requests completed")
        qs = self._get_queue_states()
        for _ in range(self.PROCESS_STEPS * 2):
            self.frames.append(self._build_frame(
                queue_states=qs,
                log_text=f"✅ All queues cleared! {len(self.requests)} requests completed"
            ))
        
        # Package
        plotly_frames = [go.Frame(data=f, name=str(i)) for i, f in enumerate(self.frames)]
        
        layout = go.Layout(
            title=dict(text='<b>Queue States: Pending → Granted → Completed</b>',
                       font=dict(size=16, color='white'), x=0.5),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-4.5, 4.5]),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False,
                       range=[-3.5, 3.5], scaleanchor='x'),
            height=650, plot_bgcolor='#1a1a2e', paper_bgcolor='#1a1a2e',
            margin=dict(l=20, r=20, t=60, b=20), showlegend=False,
            updatemenus=[dict(
                type='buttons', showactive=False, x=0.05, y=-0.02,
                xanchor='left', yanchor='top', direction='right',
                buttons=[
                    dict(label='▶ Play', method='animate',
                         args=[None, dict(frame=dict(duration=50, redraw=True),
                                         fromcurrent=True, transition=dict(duration=0))]),
                    dict(label='⏸ Pause', method='animate',
                         args=[[None], dict(frame=dict(duration=0, redraw=False),
                                           mode='immediate')])
                ]
            )],
            sliders=[dict(
                active=0, yanchor='top', xanchor='left',
                currentvalue=dict(font=dict(size=11, color='white'), prefix='Step: ', visible=True),
                transition=dict(duration=0), pad=dict(b=10, t=30),
                len=0.7, x=0.2, y=-0.02,
                steps=[dict(args=[[str(i)], dict(frame=dict(duration=0, redraw=True),
                            mode='immediate')], label='', method='animate')
                       for i in range(len(plotly_frames))]
            )]
        )
        
        return self.frames[0], plotly_frames, layout, self.logs, self.requests


# ═══════════════════════════════════════════════════════════════
#                       STREAMLIT APP
# ═══════════════════════════════════════════════════════════════

st.set_page_config(page_title='Token-Ring ME', page_icon='🔐', layout='wide')

st.markdown("""
<style>
.block-container{padding-top:0.8rem}
.hdr{font-size:1.7rem;font-weight:700;color:#0d47a1;text-align:center;padding:0.6rem;
background:linear-gradient(90deg,#e3f2fd,#bbdefb);border-radius:10px;margin-bottom:0.8rem}
.info-box{background:#e8f5e9;padding:1rem;border-radius:8px;margin:0.5rem 0}
</style>
<div class="hdr">🔐 Token-Ring Mutual Exclusion — Automated Random Scenarios</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
#                          TABS
# ══════════════════════════════════════════════════════════════

tab1, tab2, tab3, tab4 = st.tabs([
    '📡 1. Broadcasting',
    '🏆 2. Priority Granting', 
    '📋 3. Request Logs',
    '📊 4. Queue States'
])

# ══════════════════════════════════════════════════════════════
#                    TAB 1: BROADCASTING
# ══════════════════════════════════════════════════════════════

with tab1:
    st.markdown('### 📡 Request Broadcasting Animation')
    
    st.info('''
    **How it works:**
    1. Click **"Generate Random Scenario"** — this picks 3-8 random MHs with random priorities (1-10)
    2. Click **"▶ Play"** on the animation — watch each request broadcast from its MH → local MSS → ALL other MSSs
    3. By the end, every MSS holds a replicated copy of every request
    ''')
    
    col1, col2 = st.columns([1, 2])
    with col1:
        num_mss_b = st.selectbox('Number of MSSs', [4, 5, 6], index=1, key='bcast_n')
        min_req = st.slider('Min requests', 2, 6, 3, key='bcast_min')
        max_req = st.slider('Max requests', min_req + 1, 10, min(8, min_req + 4), key='bcast_max')
    
    if st.button('🎲 Generate Random Scenario', key='gen_bcast', type='primary', use_container_width=True):
        with st.spinner('Generating random requests and building animation...'):
            g = Geom(num_mss_b)
            requests = generate_random_scenario(g, min_req, max_req)
            
            st.session_state.bcast_requests = requests
            st.session_state.bcast_geom = g
            
            animator = BroadcastAnimator(g, requests)
            d0, frames, layout, logs = animator.build()
            st.session_state.bcast_anim = (d0, frames, layout, logs)
    
    if 'bcast_anim' in st.session_state:
        # Show generated requests
        st.markdown('#### 🎲 Generated Requests')
        req_df = pd.DataFrame([r.row() for r in st.session_state.bcast_requests])
        st.dataframe(req_df, use_container_width=True, hide_index=True)
        
        st.markdown('#### 📺 Animation')
        st.caption('Click **▶ Play** to watch the broadcast sequence')
        
        d0, frames, layout, logs = st.session_state.bcast_anim
        fig = go.Figure(data=d0, frames=frames, layout=layout)
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown('#### 📖 Broadcast Log')
        for log in logs:
            if log.startswith("Request"):
                st.markdown(f"**{log}**")
            else:
                st.caption(log)

# ══════════════════════════════════════════════════════════════
#                 TAB 2: PRIORITY GRANTING
# ══════════════════════════════════════════════════════════════

with tab2:
    st.markdown('### 🏆 Priority-Based Granting Animation')
    
    st.info('''
    **How it works:**
    1. Click **"Generate Random Scenario"** — creates random requests with random priorities
    2. Click **"▶ Play"** — watch the token circulate and grant access to the **highest priority** request at each MSS
    3. The bar chart on the right shows all requests sorted by priority; **green = next to be granted**
    ''')
    
    col1, col2 = st.columns([1, 2])
    with col1:
        num_mss_g = st.selectbox('Number of MSSs', [4, 5, 6], index=1, key='grant_n')
        min_req_g = st.slider('Min requests', 2, 6, 3, key='grant_min')
        max_req_g = st.slider('Max requests', min_req_g + 1, 10, min(8, min_req_g + 4), key='grant_max')
    
    if st.button('🎲 Generate Random Scenario', key='gen_grant', type='primary', use_container_width=True):
        with st.spinner('Generating random requests and building animation...'):
            g = Geom(num_mss_g)
            requests = generate_random_scenario(g, min_req_g, max_req_g)
            
            st.session_state.grant_requests = requests
            st.session_state.grant_geom = g
            
            animator = GrantingAnimator(g, requests)
            d0, frames, layout, logs, final_reqs = animator.build()
            st.session_state.grant_anim = (d0, frames, layout, logs, final_reqs)
    
    if 'grant_anim' in st.session_state:
        # Show generated requests
        st.markdown('#### 🎲 Generated Requests (sorted by priority)')
        reqs_sorted = sorted(st.session_state.grant_requests, 
                            key=lambda r: (-r.priority, r.timestamp))
        req_df = pd.DataFrame([r.row() for r in reqs_sorted])
        st.dataframe(req_df, use_container_width=True, hide_index=True)
        
        st.markdown('#### 📺 Animation')
        st.caption('Click **▶ Play** — bar chart on right shows priority queue, green = next grant')
        
        d0, frames, layout, logs, final_reqs = st.session_state.grant_anim
        fig = go.Figure(data=d0, frames=frames, layout=layout)
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown('#### 📖 Granting Log')
        for log in logs:
            if "Granting" in log or "Complete" in log:
                st.markdown(f"**{log}**")
            else:
                st.caption(log)
        
        st.markdown('#### 🏆 Granting Order Explanation')
        st.markdown('''
        The token visits each MSS in order (0 → 1 → 2 → ...). At each MSS:
        1. Check for **local** pending requests (requests that originated at this MSS)
        2. If found, grant to the one with **highest priority**
        3. If tie, grant to the one with **earliest timestamp** (first-come-first-served)
        4. If no local pending requests, pass token to next MSS
        ''')

# ══════════════════════════════════════════════════════════════
#                    TAB 3: REQUEST LOGS
# ══════════════════════════════════════════════════════════════

with tab3:
    st.markdown('### 📋 Replicated Request Logs at Each MSS')
    
    st.info('''
    **How it works:**
    1. Click **"Generate Random Scenario"** — creates random requests
    2. See how each MSS maintains a **replicated log** of ALL requests in the system
    3. Each MSS knows about requests from its own MHs AND requests broadcast from other MSSs
    ''')
    
    col1, col2 = st.columns([1, 2])
    with col1:
        num_mss_l = st.selectbox('Number of MSSs', [4, 5, 6], index=1, key='logs_n')
        min_req_l = st.slider('Min requests', 2, 6, 4, key='logs_min')
        max_req_l = st.slider('Max requests', min_req_l + 1, 12, min(10, min_req_l + 5), key='logs_max')
    
    if st.button('🎲 Generate Random Scenario', key='gen_logs', type='primary', use_container_width=True):
        with st.spinner('Generating random requests...'):
            g = Geom(num_mss_l)
            requests = generate_random_scenario(g, min_req_l, max_req_l)
            st.session_state.logs_requests = requests
            st.session_state.logs_geom = g
    
    if 'logs_requests' in st.session_state:
        requests = st.session_state.logs_requests
        g = st.session_state.logs_geom
        
        st.markdown('#### 🎲 Generated Requests')
        req_df = pd.DataFrame([r.row() for r in requests])
        st.dataframe(req_df, use_container_width=True, hide_index=True)
        
        st.markdown('---')
        st.markdown('#### 📋 Replicated Logs at Each MSS')
        st.caption('After broadcasting, EVERY MSS holds a copy of EVERY request:')
        
        # Create columns for each MSS
        cols = st.columns(g.n)
        
        for mss_id in range(g.n):
            with cols[mss_id]:
                st.markdown(f'**MSS_{mss_id}**')
                
                # All requests are replicated to all MSSs
                local = [r for r in requests if r.source_mss == mss_id]
                remote = [r for r in requests if r.source_mss != mss_id]
                
                st.success(f'📍 Local: {len(local)}')
                for r in local:
                    st.caption(f"• {r.mh_id} P={r.priority}")
                
                st.info(f'📡 Replicated: {len(remote)}')
                for r in remote:
                    st.caption(f"• {r.mh_id} P={r.priority}")
                
                st.metric("Total Entries", len(requests))
        
        st.markdown('---')
        st.markdown('#### 📊 Replication Summary')
        
        summary = []
        for mss_id in range(g.n):
            local = sum(1 for r in requests if r.source_mss == mss_id)
            remote = len(requests) - local
            summary.append({
                "MSS": f"MSS_{mss_id}",
                "Local Requests": local,
                "Replicated (from others)": remote,
                "Total Log Entries": len(requests),
                "Messages Sent": remote,  # Broadcast to each other MSS
                "Messages Received": remote
            })
        
        st.dataframe(pd.DataFrame(summary), use_container_width=True, hide_index=True)
        
        st.markdown('''
        **Key Insight:** Every MSS has the complete picture of all pending requests.
        This allows any MSS (when it receives the token) to make an informed priority-based decision.
        ''')

# ══════════════════════════════════════════════════════════════
#                   TAB 4: QUEUE STATES
# ══════════════════════════════════════════════════════════════

with tab4:
    st.markdown('### 📊 Queue States Animation')
    
    st.info('''
    **How it works:**
    1. Click **"Generate Random Scenario"** — creates random requests distributed across MSSs
    2. Click **"▶ Play"** — watch the queues at each MSS transition: **Pending → Granted → Completed**
    3. The token visits each MSS and processes requests in priority order
    ''')
    
    col1, col2 = st.columns([1, 2])
    with col1:
        num_mss_q = st.selectbox('Number of MSSs', [4, 5, 6], index=1, key='queue_n')
        min_req_q = st.slider('Min requests', 2, 6, 3, key='queue_min')
        max_req_q = st.slider('Max requests', min_req_q + 1, 10, min(8, min_req_q + 4), key='queue_max')
    
    if st.button('🎲 Generate Random Scenario', key='gen_queue', type='primary', use_container_width=True):
        with st.spinner('Generating random requests and building animation...'):
            g = Geom(num_mss_q)
            requests = generate_random_scenario(g, min_req_q, max_req_q)
            
            st.session_state.queue_requests = requests
            st.session_state.queue_geom = g
            
            animator = QueueAnimator(g, requests)
            d0, frames, layout, logs, final_reqs = animator.build()
            st.session_state.queue_anim = (d0, frames, layout, logs, final_reqs)
    
    if 'queue_anim' in st.session_state:
        # Show generated requests
        st.markdown('#### 🎲 Generated Requests')
        req_df = pd.DataFrame([r.row() for r in st.session_state.queue_requests])
        st.dataframe(req_df, use_container_width=True, hide_index=True)
        
        # Distribution chart
        st.markdown('#### 📊 Initial Queue Distribution')
        g = st.session_state.queue_geom
        dist_data = []
        for mss_id in range(g.n):
            count = sum(1 for r in st.session_state.queue_requests if r.source_mss == mss_id)
            dist_data.append({"MSS": f"MSS_{mss_id}", "Pending Requests": count})
        
        dist_fig = go.Figure(go.Bar(
            x=[d["MSS"] for d in dist_data],
            y=[d["Pending Requests"] for d in dist_data],
            marker_color=['#FF5722' if d["Pending Requests"] > 0 else '#ccc' for d in dist_data],
            text=[d["Pending Requests"] for d in dist_data],
            textposition='outside'
        ))
        dist_fig.update_layout(
            height=250, margin=dict(l=20, r=20, t=30, b=20),
            xaxis_title="", yaxis_title="Requests",
            plot_bgcolor='white'
        )
        st.plotly_chart(dist_fig, use_container_width=True)
        
        st.markdown('#### 📺 Animation')
        st.caption('Click **▶ Play** — watch queues drain as token circulates')
        
        d0, frames, layout, logs, final_reqs = st.session_state.queue_anim
        fig = go.Figure(data=d0, frames=frames, layout=layout)
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown('#### 📖 Processing Log')
        for log in logs:
            if "Serving" in log or "state" in log:
                st.markdown(f"**{log}**")
            else:
                st.caption(log)
        
        st.markdown('#### ✅ Final Queue States')
        st.success('All queues cleared — every request has been served!')
        
        final_df = pd.DataFrame([{
            "MSS": f"MSS_{mss_id}",
            "Pending": 0,
            "Completed": sum(1 for r in final_reqs if r.source_mss == mss_id)
        } for mss_id in range(g.n)])
        st.dataframe(final_df, use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════════════════════
#                        FOOTER
# ══════════════════════════════════════════════════════════════

st.markdown('---')
st.caption('Token-Ring Mutual Exclusion with Replication | Fully Automated Random Scenarios')
