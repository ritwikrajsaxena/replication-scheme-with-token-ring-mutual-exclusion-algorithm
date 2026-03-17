"""
Token-Ring Mutual Exclusion with Replication - Complete Streamlit App
=====================================================================
Demonstrates:
1. Animated token ring with MSSs and MHs
2. Request messages, token holding, handoff scenarios
3. Request broadcasting to all MSSs and priority-based granting
4. Replicated request logs and queue states
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
#                   COMPREHENSIVE ANIMATION SYSTEM
# ═══════════════════════════════════════════════════════════════

SPEED_MAP = {"🐢 Slow": 0.15, "🚶 Medium": 0.07, "🏃 Fast": 0.03, "⚡ Very Fast": 0.015}


class AnimationBuilder:
    """
    Builds smooth token ring animation with proper interpolation.
    Tracks token position explicitly to avoid jumps.
    """
    
    def __init__(self, num_mss: int, requesting_mh: str, handoff_mh: str):
        self.num_mss = num_mss
        self.requesting_mh = requesting_mh
        self.handoff_mh = handoff_mh
        
        # Geometry
        self.radius = 2.2
        self.mh_radius = 0.55
        
        # Calculate positions
        self.angles = [2 * math.pi * i / num_mss - math.pi / 2 for i in range(num_mss)]
        self.mss_x = [self.radius * math.cos(a) for a in self.angles]
        self.mss_y = [self.radius * math.sin(a) for a in self.angles]
        
        self.mh_positions = {}
        self.mh_names = {}
        for mss_id in range(num_mss):
            mss_angle = self.angles[mss_id]
            positions = []
            names = []
            for mh_idx in range(3):
                mh_angle = mss_angle + math.pi + (mh_idx - 1) * 0.4
                mx = self.mss_x[mss_id] + self.mh_radius * math.cos(mh_angle)
                my = self.mss_y[mss_id] + self.mh_radius * math.sin(mh_angle)
                positions.append((mx, my))
                names.append(f"MH_{mss_id}_{chr(65 + mh_idx)}")
            self.mh_positions[mss_id] = positions
            self.mh_names[mss_id] = names
        
        # Parse MH info
        self.req_mss_id = int(requesting_mh.split('_')[1])
        self.req_mh_idx = ord(requesting_mh.split('_')[2]) - 65
        self.handoff_mss_id = int(handoff_mh.split('_')[1])
        self.handoff_mh_idx = ord(handoff_mh.split('_')[2]) - 65
        self.handoff_target_mss = (self.handoff_mss_id + 1) % num_mss
        
        # Colors
        self.colors = {
            'mss_normal': '#00D4FF',
            'mss_pending': '#FFD700',
            'mss_holding': '#00FF00',
            'mss_granting': '#CC00FF',
            'mss_checking': '#87CEEB',
            'mh_normal': '#4CAF50',
            'mh_requesting': '#FF5722',
            'mh_in_cs': '#9C27B0',
            'mh_handoff': '#FF9800',
            'token_free': '#FFFFFF',
            'token_held': '#00FF00',
            'request_msg': '#FF5722',
            'release_msg': '#2196F3',
            'permission_msg': '#9C27B0',
            'ring': '#444444',
            'connection': '#666666',
        }
        
        # Animation storage
        self.frames_data = []
        self.phase_markers = []
        self.log_entries = []
        
        # Current token position (tracked explicitly)
        self.token_pos = 0.0
        
        # Timing
        self.move_steps = 15
        self.stop_frames = 10
        self.hold_frames = 20
        self.msg_steps = 10
    
    def lerp(self, a, b, t):
        """Linear interpolation"""
        return a + (b - a) * max(0.0, min(1.0, t))
    
    def add_phase(self, text):
        """Mark a new phase"""
        self.phase_markers.append((len(self.frames_data), text))
        self.log_entries.append(text)
    
    def build_frame(self, token_color, mss_colors, mh_colors,
                    request_msg=None, release_msg=None, permission_msg=None,
                    handoff_line=None, log_text="", mh_override_pos=None):
        """Build traces for a single frame using current token_pos"""
        
        data = []
        
        # Ring circle
        ring_angles = np.linspace(0, 2 * np.pi, 100)
        data.append(go.Scatter(
            x=[self.radius * 1.1 * math.cos(a) for a in ring_angles],
            y=[self.radius * 1.1 * math.sin(a) for a in ring_angles],
            mode='lines',
            line=dict(color=self.colors['ring'], width=3),
            hoverinfo='none', showlegend=False,
        ))
        
        # Direction arrows
        for i in range(self.num_mss):
            mid_angle = (self.angles[i] + self.angles[(i + 1) % self.num_mss]) / 2
            if i == self.num_mss - 1:
                mid_angle = (self.angles[i] + self.angles[0] + 2 * math.pi) / 2
            ax = self.radius * 1.1 * math.cos(mid_angle)
            ay = self.radius * 1.1 * math.sin(mid_angle)
            arrow_angle = mid_angle + math.pi / 2
            arrow_len = 0.12
            data.append(go.Scatter(
                x=[ax, ax + arrow_len * math.cos(arrow_angle)],
                y=[ay, ay + arrow_len * math.sin(arrow_angle)],
                mode='lines', line=dict(color=self.colors['ring'], width=2),
                hoverinfo='none', showlegend=False,
            ))
        
        # Connection lines
        for mss_id in range(self.num_mss):
            for mh_idx in range(3):
                mx, my = self.mh_positions[mss_id][mh_idx]
                mh_name = self.mh_names[mss_id][mh_idx]
                if mh_override_pos and mh_name in mh_override_pos:
                    continue
                data.append(go.Scatter(
                    x=[self.mss_x[mss_id], mx], y=[self.mss_y[mss_id], my],
                    mode='lines', line=dict(color=self.colors['connection'], width=1, dash='dot'),
                    hoverinfo='none', showlegend=False,
                ))
        
        # MSS nodes
        mss_color_list = [mss_colors.get(i, self.colors['mss_normal']) for i in range(self.num_mss)]
        mss_sizes = [55 if i in mss_colors else 48 for i in range(self.num_mss)]
        data.append(go.Scatter(
            x=self.mss_x, y=self.mss_y, mode='markers+text',
            marker=dict(size=mss_sizes, color=mss_color_list,
                        line=dict(width=3, color='white'), symbol='square'),
            text=[f'MSS_{i}' for i in range(self.num_mss)],
            textposition='top center',
            textfont=dict(color='white', size=11, family='Arial Black'),
            hoverinfo='text',
            hovertext=[f'MSS_{i}' for i in range(self.num_mss)],
            showlegend=False,
        ))
        
        # MH nodes
        for mss_id in range(self.num_mss):
            for mh_idx in range(3):
                mh_name = self.mh_names[mss_id][mh_idx]
                if mh_override_pos and mh_name in mh_override_pos:
                    mx, my = mh_override_pos[mh_name]
                else:
                    mx, my = self.mh_positions[mss_id][mh_idx]
                mh_color = mh_colors.get(mh_name, self.colors['mh_normal'])
                mh_size = 28 if mh_name in mh_colors else 22
                data.append(go.Scatter(
                    x=[mx], y=[my], mode='markers+text',
                    marker=dict(size=mh_size, color=mh_color,
                                line=dict(width=2, color='white'), symbol='circle'),
                    text=[mh_name.split('_')[-1]],
                    textposition='bottom center',
                    textfont=dict(color='white', size=10),
                    hoverinfo='text', hovertext=mh_name, showlegend=False,
                ))
        
        # Messages
        if request_msg:
            data.append(go.Scatter(
                x=[request_msg['x']], y=[request_msg['y']],
                mode='markers+text',
                marker=dict(size=15, color=self.colors['request_msg'],
                            symbol='square', line=dict(width=2, color='white')),
                text=['REQ'], textposition='top center',
                textfont=dict(color='white', size=8),
                hoverinfo='text', hovertext='REQUEST', showlegend=False,
            ))
        
        if permission_msg:
            data.append(go.Scatter(
                x=[permission_msg['x']], y=[permission_msg['y']],
                mode='markers+text',
                marker=dict(size=15, color=self.colors['permission_msg'],
                            symbol='square', line=dict(width=2, color='white')),
                text=['PERM'], textposition='top center',
                textfont=dict(color='white', size=8),
                hoverinfo='text', hovertext='PERMISSION', showlegend=False,
            ))
        
        if release_msg:
            data.append(go.Scatter(
                x=[release_msg['x']], y=[release_msg['y']],
                mode='markers+text',
                marker=dict(size=15, color=self.colors['release_msg'],
                            symbol='square', line=dict(width=2, color='white')),
                text=['REL'], textposition='top center',
                textfont=dict(color='white', size=8),
                hoverinfo='text', hovertext='RELEASE', showlegend=False,
            ))
        
        if handoff_line:
            data.append(go.Scatter(
                x=[handoff_line['x1'], handoff_line['x2']],
                y=[handoff_line['y1'], handoff_line['y2']],
                mode='lines', line=dict(color=self.colors['mh_handoff'], width=3, dash='dash'),
                hoverinfo='none', showlegend=False,
            ))
        
        # Token (use current self.token_pos)
        token_angle = (self.token_pos / self.num_mss) * 2 * math.pi - math.pi / 2
        tx = self.radius * math.cos(token_angle)
        ty = self.radius * math.sin(token_angle)
        data.append(go.Scatter(
            x=[tx], y=[ty], mode='markers+text',
            marker=dict(size=35, color=token_color, symbol='circle',
                        line=dict(width=4, color='#333')),
            text=['🔑'], textfont=dict(size=14),
            hoverinfo='text', hovertext='TOKEN', showlegend=False,
        ))
        
        # Log text
        data.append(go.Scatter(
            x=[0], y=[-3.3], mode='text',
            text=[f'<b>{log_text}</b>'],
            textfont=dict(size=13, color='white'),
            hoverinfo='none', showlegend=False,
        ))
        
        return data
    
    def add_frame(self, **kwargs):
        """Add a frame to the animation"""
        self.frames_data.append(self.build_frame(**kwargs))
    
    def move_token_to(self, target_pos, token_color, mss_colors, mh_colors, log_text, **extra):
        """Smoothly move token from current position to target position"""
        start_pos = self.token_pos
        for step in range(self.move_steps):
            t = (step + 1) / self.move_steps
            self.token_pos = self.lerp(start_pos, target_pos, t)
            self.add_frame(token_color=token_color, mss_colors=mss_colors,
                          mh_colors=mh_colors, log_text=log_text, **extra)
    
    def hold_token(self, frames, token_color, mss_colors, mh_colors, log_text, **extra):
        """Hold token at current position for N frames"""
        for _ in range(frames):
            self.add_frame(token_color=token_color, mss_colors=mss_colors,
                          mh_colors=mh_colors, log_text=log_text, **extra)
    
    def send_message(self, from_pos, to_pos, msg_type, token_color, mss_colors, mh_colors, log_text, **extra):
        """Animate a message traveling between two points"""
        fx, fy = from_pos
        tx, ty = to_pos
        for step in range(self.msg_steps):
            t = (step + 1) / self.msg_steps
            mx = self.lerp(fx, tx, t)
            my = self.lerp(fy, ty, t)
            
            msg_dict = {msg_type: {'x': mx, 'y': my}}
            self.add_frame(token_color=token_color, mss_colors=mss_colors,
                          mh_colors=mh_colors, log_text=log_text, **msg_dict, **extra)
    
    def build(self):
        """Build the complete animation"""
        
        req_mh = self.requesting_mh
        req_mss = self.req_mss_id
        mh_x, mh_y = self.mh_positions[req_mss][self.req_mh_idx]
        mss_x, mss_y = self.mss_x[req_mss], self.mss_y[req_mss]
        
        # ══════════════════════════════════════════════════════════════
        # PHASE 1: MH sends REQUEST (token still at MSS_0)
        # ══════════════════════════════════════════════════════════════
        self.add_phase(f"Phase 1: {req_mh} sends REQUEST to MSS_{req_mss}")
        self.token_pos = 0.0
        
        self.send_message(
            from_pos=(mh_x, mh_y), to_pos=(mss_x, mss_y),
            msg_type='request_msg',
            token_color=self.colors['token_free'],
            mss_colors={},
            mh_colors={req_mh: self.colors['mh_requesting']},
            log_text=f"📤 {req_mh} sending REQUEST to MSS_{req_mss}"
        )
        
        # ══════════════════════════════════════════════════════════════
        # PHASE 2: MSS queues request
        # ══════════════════════════════════════════════════════════════
        self.add_phase(f"Phase 2: MSS_{req_mss} queues request")
        
        self.hold_token(
            frames=self.stop_frames,
            token_color=self.colors['token_free'],
            mss_colors={req_mss: self.colors['mss_pending']},
            mh_colors={req_mh: self.colors['mh_requesting']},
            log_text=f"📋 MSS_{req_mss} queued request from {req_mh}"
        )
        
        # ══════════════════════════════════════════════════════════════
        # PHASE 3: Token circulates, checking each MSS
        # ══════════════════════════════════════════════════════════════
        self.add_phase("Phase 3: Token circulates, checking each MSS")
        
        # Visit each MSS from 0 up to (but not including) req_mss
        for target_mss in range(req_mss):
            if target_mss > 0:
                # Move to this MSS
                self.move_token_to(
                    target_pos=target_mss,
                    token_color=self.colors['token_free'],
                    mss_colors={req_mss: self.colors['mss_pending']},
                    mh_colors={req_mh: self.colors['mh_requesting']},
                    log_text=f"⚪ Token moving to MSS_{target_mss}…"
                )
            
            # Brief stop to check
            self.hold_token(
                frames=self.stop_frames,
                token_color=self.colors['token_free'],
                mss_colors={target_mss: self.colors['mss_checking'],
                           req_mss: self.colors['mss_pending']},
                mh_colors={req_mh: self.colors['mh_requesting']},
                log_text=f"🔍 MSS_{target_mss}: No pending requests → passing token"
            )
        
        # ══════════════════════════════════════════════════════════════
        # PHASE 4: Token arrives at requesting MSS
        # ══════════════════════════════════════════════════════════════
        self.add_phase(f"Phase 4: Token arrives at MSS_{req_mss}")
        
        self.move_token_to(
            target_pos=req_mss,
            token_color=self.colors['token_free'],
            mss_colors={req_mss: self.colors['mss_pending']},
            mh_colors={req_mh: self.colors['mh_requesting']},
            log_text=f"⚪ Token arriving at MSS_{req_mss}…"
        )
        
        # ══════════════════════════════════════════════════════════════
        # PHASE 5: MSS HOLDS token
        # ══════════════════════════════════════════════════════════════
        self.add_phase(f"Phase 5: MSS_{req_mss} HOLDS token")
        
        self.hold_token(
            frames=self.hold_frames,
            token_color=self.colors['token_held'],
            mss_colors={req_mss: self.colors['mss_holding']},
            mh_colors={req_mh: self.colors['mh_requesting']},
            log_text=f"🟢 MSS_{req_mss} HOLDING token! Pending request found."
        )
        
        # ══════════════════════════════════════════════════════════════
        # PHASE 6: Grant PERMISSION
        # ══════════════════════════════════════════════════════════════
        self.add_phase(f"Phase 6: MSS_{req_mss} grants PERMISSION")
        
        self.send_message(
            from_pos=(mss_x, mss_y), to_pos=(mh_x, mh_y),
            msg_type='permission_msg',
            token_color=self.colors['token_held'],
            mss_colors={req_mss: self.colors['mss_granting']},
            mh_colors={req_mh: self.colors['mh_requesting']},
            log_text=f"📨 MSS_{req_mss} sending PERMISSION to {req_mh}"
        )
        
        # ══════════════════════════════════════════════════════════════
        # PHASE 7: MH in Critical Section
        # ══════════════════════════════════════════════════════════════
        self.add_phase(f"Phase 7: {req_mh} in CRITICAL SECTION")
        
        self.hold_token(
            frames=self.hold_frames,
            token_color=self.colors['token_held'],
            mss_colors={req_mss: self.colors['mss_holding']},
            mh_colors={req_mh: self.colors['mh_in_cs']},
            log_text=f"🟣 {req_mh} executing in CRITICAL SECTION…"
        )
        
        # ══════════════════════════════════════════════════════════════
        # PHASE 8: MH sends RELEASE
        # ══════════════════════════════════════════════════════════════
        self.add_phase(f"Phase 8: {req_mh} sends RELEASE")
        
        self.send_message(
            from_pos=(mh_x, mh_y), to_pos=(mss_x, mss_y),
            msg_type='release_msg',
            token_color=self.colors['token_held'],
            mss_colors={req_mss: self.colors['mss_holding']},
            mh_colors={req_mh: self.colors['mh_in_cs']},
            log_text=f"📤 {req_mh} sending RELEASE to MSS_{req_mss}"
        )
        
        # ══════════════════════════════════════════════════════════════
        # PHASE 9: Handoff MH sends request
        # ══════════════════════════════════════════════════════════════
        ho_mh = self.handoff_mh
        ho_mss = self.handoff_mss_id
        ho_mh_x, ho_mh_y = self.mh_positions[ho_mss][self.handoff_mh_idx]
        ho_mss_x, ho_mss_y = self.mss_x[ho_mss], self.mss_y[ho_mss]
        
        self.add_phase(f"Phase 9: Token released; {ho_mh} sends request")
        
        # Token starts moving, handoff MH sends request
        self.send_message(
            from_pos=(ho_mh_x, ho_mh_y), to_pos=(ho_mss_x, ho_mss_y),
            msg_type='request_msg',
            token_color=self.colors['token_free'],
            mss_colors={},
            mh_colors={ho_mh: self.colors['mh_requesting']},
            log_text=f"📤 {ho_mh} sending REQUEST to MSS_{ho_mss}"
        )
        
        self.hold_token(
            frames=self.stop_frames,
            token_color=self.colors['token_free'],
            mss_colors={ho_mss: self.colors['mss_pending']},
            mh_colors={ho_mh: self.colors['mh_requesting']},
            log_text=f"📋 MSS_{ho_mss} queued request from {ho_mh}"
        )
        
        # ══════════════════════════════════════════════════════════════
        # PHASE 10: HANDOFF - MH moves to new MSS
        # ══════════════════════════════════════════════════════════════
        new_mss = self.handoff_target_mss
        new_mh_x, new_mh_y = self.mh_positions[new_mss][0]
        
        self.add_phase(f"Phase 10: HANDOFF! {ho_mh} moves to MSS_{new_mss}")
        
        for step in range(self.msg_steps * 2):
            t = (step + 1) / (self.msg_steps * 2)
            cx = self.lerp(ho_mh_x, new_mh_x, t)
            cy = self.lerp(ho_mh_y, new_mh_y, t)
            
            # Token also moves slightly
            self.token_pos = self.lerp(req_mss, req_mss + 0.5, t)
            
            self.add_frame(
                token_color=self.colors['token_free'],
                mss_colors={ho_mss: self.colors['mss_pending']},
                mh_colors={ho_mh: self.colors['mh_handoff']},
                handoff_line={'x1': ho_mh_x, 'y1': ho_mh_y, 'x2': cx, 'y2': cy},
                log_text=f"📱 HANDOFF: {ho_mh} moving to MSS_{new_mss}…",
                mh_override_pos={ho_mh: (cx, cy)}
            )
        
        # ══════════════════════════════════════════════════════════════
        # PHASE 11: Request KILLED
        # ══════════════════════════════════════════════════════════════
        self.add_phase(f"Phase 11: Request KILLED at MSS_{ho_mss}")
        
        for step in range(self.hold_frames):
            mss_col = '#FF0000' if step % 6 < 3 else self.colors['mss_normal']
            self.add_frame(
                token_color=self.colors['token_free'],
                mss_colors={ho_mss: mss_col},
                mh_colors={},
                log_text=f"❌ Request from {ho_mh} KILLED! MH no longer at MSS_{ho_mss}",
                mh_override_pos={ho_mh: (new_mh_x, new_mh_y)}
            )
        
        # ══════════════════════════════════════════════════════════════
        # PHASE 12: MH re-registers at new MSS
        # ══════════════════════════════════════════════════════════════
        new_mss_x, new_mss_y = self.mss_x[new_mss], self.mss_y[new_mss]
        
        self.add_phase(f"Phase 12: {ho_mh} re-registers at MSS_{new_mss}")
        
        for step in range(self.msg_steps):
            t = (step + 1) / self.msg_steps
            mx = self.lerp(new_mh_x, new_mss_x, t)
            my = self.lerp(new_mh_y, new_mss_y, t)
            
            self.add_frame(
                token_color=self.colors['token_free'],
                mss_colors={},
                mh_colors={ho_mh: self.colors['mh_requesting']},
                request_msg={'x': mx, 'y': my},
                log_text=f"📤 {ho_mh} RE-REGISTERING at MSS_{new_mss}",
                mh_override_pos={ho_mh: (new_mh_x, new_mh_y)}
            )
        
        self.hold_token(
            frames=self.stop_frames,
            token_color=self.colors['token_free'],
            mss_colors={new_mss: self.colors['mss_pending']},
            mh_colors={ho_mh: self.colors['mh_requesting']},
            log_text=f"📋 MSS_{new_mss} queued NEW request from {ho_mh}",
            mh_override_pos={ho_mh: (new_mh_x, new_mh_y)}
        )
        
        # ══════════════════════════════════════════════════════════════
        # PHASE 13: Token continues to new MSS
        # ══════════════════════════════════════════════════════════════
        self.add_phase("Phase 13: Token continues to new MSS")
        
        # Calculate intermediate MSSs to visit
        current = int(self.token_pos) + 1
        while current % self.num_mss != new_mss:
            target = current % self.num_mss
            
            self.move_token_to(
                target_pos=target,
                token_color=self.colors['token_free'],
                mss_colors={new_mss: self.colors['mss_pending']},
                mh_colors={ho_mh: self.colors['mh_requesting']},
                log_text=f"⚪ Token moving to MSS_{target}…",
                mh_override_pos={ho_mh: (new_mh_x, new_mh_y)}
            )
            
            self.hold_token(
                frames=self.stop_frames,
                token_color=self.colors['token_free'],
                mss_colors={target: self.colors['mss_checking'],
                           new_mss: self.colors['mss_pending']},
                mh_colors={ho_mh: self.colors['mh_requesting']},
                log_text=f"🔍 MSS_{target}: No pending requests → passing token",
                mh_override_pos={ho_mh: (new_mh_x, new_mh_y)}
            )
            
            current += 1
        
        # ══════════════════════════════════════════════════════════════
        # PHASE 14: Token grants to handoff MH at new MSS
        # ══════════════════════════════════════════════════════════════
        self.add_phase(f"Phase 14: Token grants to {ho_mh} at MSS_{new_mss}")
        
        self.move_token_to(
            target_pos=new_mss,
            token_color=self.colors['token_free'],
            mss_colors={new_mss: self.colors['mss_pending']},
            mh_colors={ho_mh: self.colors['mh_requesting']},
            log_text=f"⚪ Token arriving at MSS_{new_mss}…",
            mh_override_pos={ho_mh: (new_mh_x, new_mh_y)}
        )
        
        self.hold_token(
            frames=self.hold_frames,
            token_color=self.colors['token_held'],
            mss_colors={new_mss: self.colors['mss_holding']},
            mh_colors={ho_mh: self.colors['mh_requesting']},
            log_text=f"🟢 MSS_{new_mss} HOLDING token!",
            mh_override_pos={ho_mh: (new_mh_x, new_mh_y)}
        )
        
        self.send_message(
            from_pos=(new_mss_x, new_mss_y), to_pos=(new_mh_x, new_mh_y),
            msg_type='permission_msg',
            token_color=self.colors['token_held'],
            mss_colors={new_mss: self.colors['mss_granting']},
            mh_colors={ho_mh: self.colors['mh_requesting']},
            log_text=f"📨 MSS_{new_mss} granting PERMISSION",
            mh_override_pos={ho_mh: (new_mh_x, new_mh_y)}
        )
        
        self.hold_token(
            frames=self.hold_frames,
            token_color=self.colors['token_held'],
            mss_colors={new_mss: self.colors['mss_holding']},
            mh_colors={ho_mh: self.colors['mh_in_cs']},
            log_text=f"🟣 {ho_mh} in CRITICAL SECTION (at new MSS)",
            mh_override_pos={ho_mh: (new_mh_x, new_mh_y)}
        )
        
        # ══════════════════════════════════════════════════════════════
        # PHASE 15: Normal operation
        # ══════════════════════════════════════════════════════════════
        self.add_phase("Phase 15: Normal operation resumes")
        
        for step in range(self.move_steps * 2):
            t = (step + 1) / (self.move_steps * 2)
            self.token_pos = new_mss + t * 2
            self.add_frame(
                token_color=self.colors['token_free'],
                mss_colors={}, mh_colors={},
                log_text="⚪ Normal operation: free token circulating…"
            )
        
        # Build layout
        layout = go.Layout(
            title=dict(
                text='<b>Token Ring Mutual Exclusion — MSS / MH Architecture</b>',
                font=dict(color='white', size=18), x=0.5,
            ),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False,
                       range=[-3.5, 3.5]),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False,
                       range=[-3.8, 3.5], scaleanchor='x'),
            plot_bgcolor='#1a1a2e', paper_bgcolor='#1a1a2e',
            height=700,
            margin=dict(l=20, r=20, t=60, b=60),
            showlegend=False,
        )
        
        return self.frames_data, layout, self.phase_markers, self.log_entries


def create_static_ring_with_mhs(mss_list):
    """Create a static ring visualization with MHs"""
    n = len(mss_list)
    radius = 2.0
    mh_radius = 0.5
    angles = [2 * math.pi * i / n - math.pi / 2 for i in range(n)]

    xs = [radius * math.cos(a) for a in angles]
    ys = [radius * math.sin(a) for a in angles]

    fig = go.Figure()

    ring_angles = np.linspace(0, 2 * np.pi, 100)
    fig.add_trace(go.Scatter(
        x=[radius * 1.05 * math.cos(a) for a in ring_angles],
        y=[radius * 1.05 * math.sin(a) for a in ring_angles],
        mode='lines', line=dict(color='#555', width=2),
        hoverinfo='none', showlegend=False,
    ))

    colors_list = ['gold' if mss.has_token else '#64b5f6' for mss in mss_list]
    sizes = [55 if mss.has_token else 45 for mss in mss_list]

    fig.add_trace(go.Scatter(
        x=xs, y=ys, mode='markers+text',
        text=[f'MSS_{mss.id}' for mss in mss_list],
        textposition='top center', textfont=dict(size=12, color='black'),
        marker=dict(size=sizes, color=colors_list,
                    line=dict(width=2, color='black'), symbol='square'),
        hoverinfo='text',
        hovertext=[f'MSS_{mss.id}' for mss in mss_list],
        showlegend=False,
    ))

    for idx, mss in enumerate(mss_list):
        for j, mh in enumerate(mss.mobile_hosts):
            offset_angle = angles[idx] + math.pi + (j - 1) * 0.4
            mx = xs[idx] + mh_radius * math.cos(offset_angle)
            my = ys[idx] + mh_radius * math.sin(offset_angle)

            fig.add_trace(go.Scatter(
                x=[xs[idx], mx], y=[ys[idx], my],
                mode='lines', line=dict(width=1, color='#aaa', dash='dot'),
                hoverinfo='none', showlegend=False,
            ))
            fig.add_trace(go.Scatter(
                x=[mx], y=[my], mode='markers+text',
                text=[mh.id.split('_')[-1]],
                textposition='bottom center',
                textfont=dict(size=9, color='#333'),
                marker=dict(size=18, color='#4CAF50',
                            line=dict(width=1, color='white')),
                hoverinfo='text', hovertext=f'{mh.id} | P={mh.base_priority}',
                showlegend=False,
            ))

    fig.update_layout(
        title=dict(text='<b>MSS-MH Token Ring</b>', font=dict(size=16)),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False,
                   range=[-3.5, 3.5]),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False,
                   range=[-3.5, 3.5], scaleanchor='x'),
        height=550, plot_bgcolor='white', paper_bgcolor='white',
        margin=dict(l=20, r=20, t=50, b=20),
    )
    return fig


# ═══════════════════════════════════════════════════════════════
#                    SCENARIO BUILDER
# ═══════════════════════════════════════════════════════════════

def build_scenario_with_3mhs(num_mss=4):
    ring = RingTopology(num_mss)
    mhs_list = []
    for mss_id in range(num_mss):
        for mh_idx in range(3):
            mh_name = f"MH_{mss_id}_{chr(65 + mh_idx)}"
            priority = 5 + mh_idx + mss_id
            mh = MobileHost(mh_name, ring.nodes[mss_id], base_priority=priority)
            ring.nodes[mss_id].add_mh(mh)
            mhs_list.append(mh)
    return ring, mhs_list


# ═══════════════════════════════════════════════════════════════
#                       STREAMLIT APP
# ═══════════════════════════════════════════════════════════════

st.set_page_config(page_title='Token-Ring ME Replication', page_icon='🔐', layout='wide')

st.markdown("""
<style>
.block-container{padding-top:1rem;}
.header{font-size:2rem;font-weight:700;color:#0d47a1;text-align:center;
        padding:.8rem;background:linear-gradient(90deg,#e3f2fd,#bbdefb);
        border-radius:10px;margin-bottom:1.2rem;}
.sub{color:#555;text-align:center;margin-bottom:1.5rem;}
</style>
<div class="header">🔐 Token-Ring Mutual Exclusion — Replication Scheme</div>
<p class="sub">MSS-MH Architecture &nbsp;|&nbsp; Request Broadcasting &nbsp;|&nbsp;
Priority-Based Granting &nbsp;|&nbsp; Handoff Handling</p>
""", unsafe_allow_html=True)

# ── Session state defaults ──
if 'ring' not in st.session_state:
    r, m = build_scenario_with_3mhs(4)
    st.session_state.ring = r
    st.session_state.mhs = m
    st.session_state.tm = TokenManager(r)
    st.session_state.step = 0
    st.session_state.reqs_made = []

_defaults = {
    'anim_frames': None, 'anim_layout': None,
    'anim_phases': [], 'anim_log_entries': [],
    'anim_frame': 0, 'anim_speed': 0.07,
}
for k, v in _defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

ring: RingTopology = st.session_state.ring
mhs: List[MobileHost] = st.session_state.mhs
tm: TokenManager = st.session_state.tm

# ── Sidebar ──
with st.sidebar:
    st.header('⚙️ Controls')

    if st.button('🔄 Reset Everything', use_container_width=True):
        for k in list(st.session_state.keys()):
            del st.session_state[k]
        st.rerun()

    st.markdown('---')
    st.subheader('📤 Send Request')

    mh_by_mss = {}
    for mh in mhs:
        mh_by_mss.setdefault(mh.current_mss.id, []).append(mh)

    selected_mss = st.selectbox('MSS', range(ring.n), format_func=lambda i: f'MSS_{i}')
    mhs_at_mss = mh_by_mss.get(selected_mss, [])

    if mhs_at_mss:
        selected_mh = st.selectbox(
            'MH', range(len(mhs_at_mss)),
            format_func=lambda i: f'{mhs_at_mss[i].id} (P={mhs_at_mss[i].base_priority})')

        if st.button('📤 Send Request', use_container_width=True):
            req = mhs_at_mss[selected_mh].request_cs()
            if req:
                st.session_state.reqs_made.append(req)
                st.success(f'{mhs_at_mss[selected_mh].id} requested')

    st.markdown('---')
    st.subheader('🔄 Token')

    c1, c2 = st.columns(2)
    with c1:
        if st.button('▶ Step', use_container_width=True):
            tm.step()
            st.session_state.step += 1
    with c2:
        if st.button('⏩ ×5', use_container_width=True):
            for _ in range(5):
                tm.step()
                st.session_state.step += 1

    st.markdown('---')
    holder = ring.token_holder()
    st.metric('Steps', st.session_state.step)
    st.metric('Token At', f'MSS_{holder.id}' if holder else '—')


# ════════════════════ TABS ════════════════════
tab1, tab2, tab3, tab4 = st.tabs([
    '🎬 1. Animation', '📡 2. Propagation', '📋 3. Logs', '📊 4. Queues',
])

# ────────── TAB 1: ANIMATION ──────────
with tab1:
    st.markdown('### Token Ring Animation Demo')
    st.info('''
    **Demo Scenario:** This animation demonstrates:
    1. **MH sends REQUEST** to its MSS **before** token arrives
    2. Token **smoothly moves** and **briefly stops** at each MSS
    3. If no pending requests → token passes quickly
    4. If pending request found → MSS **HOLDS** token longer
    5. **Handoff scenario:** MH moves to new MSS, old request killed

    Click **Generate Animation** then **▶ Play** to watch smoothly!
    ''')

    # ── Configuration ──
    st.markdown('#### Configuration')
    col1, col2, col3 = st.columns(3)
    with col1:
        num_mss_anim = st.selectbox('Number of MSSs', [4, 5, 6], index=2, key='num_mss')
    with col2:
        mh_options = [f"MH_{mid}_{chr(65 + j)}"
                      for mid in range(num_mss_anim) for j in range(3)]
        req_default = min(6, len(mh_options) - 1)
        requesting_mh = st.selectbox('Requesting MH', mh_options,
                                     index=req_default, key='req_mh')
    with col3:
        handoff_options = [m for m in mh_options if m != requesting_mh]
        ho_default = min(4, len(handoff_options) - 1)
        handoff_mh = st.selectbox('Handoff MH', handoff_options,
                                  index=ho_default, key='handoff_mh')

    if st.button('🎬 Generate Animation', use_container_width=True, type='primary'):
        with st.spinner('Generating frames…'):
            builder = AnimationBuilder(num_mss_anim, requesting_mh, handoff_mh)
            fd, ly, pm, le = builder.build()
            st.session_state.anim_frames = fd
            st.session_state.anim_layout = ly
            st.session_state.anim_phases = pm
            st.session_state.anim_log_entries = le
            st.session_state.anim_frame = 0
        st.rerun()

    st.markdown('---')

    # ── Playback ──
    if st.session_state.anim_frames is not None:
        total = len(st.session_state.anim_frames)

        st.markdown('#### Playback Controls')

        # Speed selector
        speed_label = st.select_slider(
            'Speed',
            options=list(SPEED_MAP.keys()),
            value='🚶 Medium',
            key='speed_slider',
        )
        st.session_state.anim_speed = SPEED_MAP[speed_label]

        # Control buttons
        ctrl1, ctrl2, ctrl3, ctrl4, ctrl5 = st.columns(5)
        with ctrl1:
            start_btn = st.button('⏮️ Start', use_container_width=True)
        with ctrl2:
            prev_btn = st.button('◀️ Prev', use_container_width=True)
        with ctrl3:
            play_btn = st.button('▶️ Play', use_container_width=True, type='primary')
        with ctrl4:
            next_btn = st.button('▶️ Next', use_container_width=True)
        with ctrl5:
            end_btn = st.button('⏭️ End', use_container_width=True)

        # Handle button actions
        if start_btn:
            st.session_state.anim_frame = 0
        if prev_btn:
            st.session_state.anim_frame = max(0, st.session_state.anim_frame - 1)
        if next_btn:
            st.session_state.anim_frame = min(total - 1, st.session_state.anim_frame + 1)
        if end_btn:
            st.session_state.anim_frame = total - 1

        # Frame slider
        new_frame = st.slider('Frame', 0, total - 1,
                              st.session_state.anim_frame,
                              key='frame_slider')
        st.session_state.anim_frame = new_frame

        # Current phase
        current_phase = ""
        for pidx, ptxt in st.session_state.anim_phases:
            if st.session_state.anim_frame >= pidx:
                current_phase = ptxt
        if current_phase:
            st.markdown(f'**🔶 {current_phase}**')

        st.caption(f'Frame {st.session_state.anim_frame + 1} / {total}')

        # Chart placeholder - single container for smooth updates
        chart_placeholder = st.empty()

        # If Play was clicked, run smooth animation loop
        if play_btn:
            for frame_idx in range(st.session_state.anim_frame, total):
                # Update phase display
                for pidx, ptxt in st.session_state.anim_phases:
                    if frame_idx >= pidx:
                        current_phase = ptxt

                # Render frame
                frame_traces = st.session_state.anim_frames[frame_idx]
                fig = go.Figure(data=frame_traces, layout=st.session_state.anim_layout)
                chart_placeholder.plotly_chart(fig, use_container_width=True, key=f'anim_{frame_idx}')

                st.session_state.anim_frame = frame_idx
                time.sleep(st.session_state.anim_speed)

            st.toast('✅ Animation complete!')
        else:
            # Show current static frame
            frame_traces = st.session_state.anim_frames[st.session_state.anim_frame]
            fig = go.Figure(data=frame_traces, layout=st.session_state.anim_layout)
            chart_placeholder.plotly_chart(fig, use_container_width=True, key='anim_static')

    else:
        st.warning('👆 Click **Generate Animation** to begin.')

    # ── Legend ──
    st.markdown('---')
    st.markdown('#### Color Legend')
    lc1, lc2, lc3 = st.columns(3)
    with lc1:
        st.markdown('''
        **Token:**
        - ⚪ White = Free
        - 🟢 Green = Held
        ''')
    with lc2:
        st.markdown('''
        **MSS:**
        - 🔵 Cyan = Normal
        - 🟡 Yellow = Pending request
        - 🟢 Green = Holding token
        - 🟣 Purple = Granting
        - 🔴 Red = Request killed
        ''')
    with lc3:
        st.markdown('''
        **MH:**
        - 🟢 Green = Normal
        - 🟠 Orange = Requesting
        - 🟣 Purple = In CS
        - 🟠 Dashed = Handoff
        ''')

    if st.session_state.anim_log_entries:
        st.markdown('---')
        st.markdown('#### Animation Phases')
        for i, entry in enumerate(st.session_state.anim_log_entries, 1):
            st.markdown(f'{i}. {entry}')


# ────────── TAB 2: PROPAGATION ──────────
with tab2:
    st.markdown('### Request Broadcasting')

    static_fig = create_static_ring_with_mhs(ring.nodes)
    st.plotly_chart(static_fig, use_container_width=True)

    if st.session_state.reqs_made:
        for req in st.session_state.reqs_made:
            with st.expander(f'{req.request_id} | {req.status}'):
                st.write(f'**MH:** {req.mh_id}')
                st.write(f'**MSS:** MSS_{req.source_mss_id}')
                st.write(f'**Priority:** {req.priority}')
    else:
        st.info('No requests yet')

    st.markdown('#### Event Log')
    if tm.event_log:
        for ev in tm.event_log[-10:]:
            st.text(ev)


# ────────── TAB 3: LOGS ──────────
with tab3:
    st.markdown('### Replicated Request Logs')

    for mss in ring.nodes:
        tok = ' 🔑' if mss.has_token else ''
        with st.expander(f'MSS_{mss.id}{tok} — {len(mss.replicated_log)} entries',
                         expanded=True):
            if mss.replicated_log:
                df = pd.DataFrame([r.to_dict() for r in mss.replicated_log])
                st.dataframe(df, use_container_width=True, hide_index=True)
            else:
                st.caption('Empty')


# ────────── TAB 4: QUEUES ──────────
with tab4:
    st.markdown('### Queue States')

    cols = st.columns(ring.n)
    for i, mss in enumerate(ring.nodes):
        with cols[i]:
            tok = ' 🔑' if mss.has_token else ''
            st.markdown(f'**MSS_{mss.id}{tok}**')
            if mss.local_queue:
                st.dataframe(pd.DataFrame([{
                    'MH': r.mh_id, 'P': r.priority, 'Status': r.status
                } for r in mss.local_queue]), hide_index=True)
            else:
                st.success('Empty ✓')

    st.markdown('---')
    st.dataframe(pd.DataFrame([mss.stats() for mss in ring.nodes]),
                 use_container_width=True, hide_index=True)


# Footer
st.markdown('---')
st.caption('Token-Ring Mutual Exclusion with Replication | MSS-MH Architecture')
