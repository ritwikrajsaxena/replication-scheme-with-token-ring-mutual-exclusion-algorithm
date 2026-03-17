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
from typing import List, Optional, Tuple, Dict
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

def create_comprehensive_animation(num_mss: int = 6, requesting_mh: str = "MH_1_A", 
                                    handoff_mh: str = "MH_2_B"):
    """
    Create a comprehensive animation showing the full token ring lifecycle:
    1. Free token circulates
    2. MH sends request to MSS (small square message)
    3. MSS queues request and changes color
    4. Token arrives, MSS holds it
    5. MSS grants permission to MH
    6. MH enters CS, then releases
    7. Another MH requests but does handoff before token arrives
    8. Request is killed, MH re-registers at new MSS
    """
    
    radius = 2.2
    mh_radius = 0.5  # Distance of MHs from their MSS
    
    # Calculate MSS positions
    angles = [2 * math.pi * i / num_mss - math.pi / 2 for i in range(num_mss)]
    mss_x = [radius * math.cos(a) for a in angles]
    mss_y = [radius * math.sin(a) for a in angles]
    
    # Calculate MH positions (3 MHs per MSS)
    mh_positions = {}  # {mss_id: [(x1,y1), (x2,y2), (x3,y3)]}
    mh_names = {}  # {mss_id: [name1, name2, name3]}
    
    for mss_id in range(num_mss):
        mss_angle = angles[mss_id]
        positions = []
        names = []
        for mh_idx in range(3):
            # Position MHs in an arc outside the MSS
            mh_angle = mss_angle + math.pi + (mh_idx - 1) * 0.4
            mx = mss_x[mss_id] + mh_radius * math.cos(mh_angle)
            my = mss_y[mss_id] + mh_radius * math.sin(mh_angle)
            positions.append((mx, my))
            names.append(f"MH_{mss_id}_{chr(65 + mh_idx)}")  # MH_0_A, MH_0_B, MH_0_C, etc.
        mh_positions[mss_id] = positions
        mh_names[mss_id] = names
    
    # Parse requesting MH info
    req_mss_id = int(requesting_mh.split('_')[1])
    req_mh_idx = ord(requesting_mh.split('_')[2]) - 65
    
    # Parse handoff MH info
    handoff_mss_id = int(handoff_mh.split('_')[1])
    handoff_mh_idx = ord(handoff_mh.split('_')[2]) - 65
    handoff_target_mss = (handoff_mss_id + 1) % num_mss  # Moves to next MSS
    
    # Colors
    colors = {
        'mss_normal': '#00D4FF',
        'mss_pending': '#FFD700',
        'mss_holding': '#00FF00',
        'mss_granting': '#CC00FF',
        'mh_normal': '#4CAF50',
        'mh_requesting': '#FF5722',
        'mh_in_cs': '#9C27B0',
        'mh_handoff': '#FF9800',
        'token_free': '#FFFFFF',
        'token_held': '#00FF00',
        'request_msg': '#FF5722',
        'release_msg': '#2196F3',
        'ring': '#444444',
        'connection': '#666666',
    }
    
    frames = []
    log_entries = []
    
    # Animation parameters
    steps_between_mss = 15
    wait_steps = 20
    message_steps = 10
    
    def interpolate(start, end, t):
        """Linear interpolation between two points"""
        return start + (end - start) * t
    
    def create_frame(token_pos, token_color, mss_colors, mh_colors, 
                     request_msg=None, release_msg=None, handoff_line=None,
                     frame_num=0, active_mh=None, log_text=""):
        """Create a single animation frame"""
        
        data = []
        
        # Ring circle
        ring_angles = np.linspace(0, 2 * np.pi, 100)
        data.append(go.Scatter(
            x=[radius * 1.1 * math.cos(a) for a in ring_angles],
            y=[radius * 1.1 * math.sin(a) for a in ring_angles],
            mode='lines',
            line=dict(color=colors['ring'], width=3),
            hoverinfo='none',
            showlegend=False,
        ))
        
        # Direction arrows on ring
        for i in range(num_mss):
            mid_angle = (angles[i] + angles[(i + 1) % num_mss]) / 2
            if i == num_mss - 1:
                mid_angle = (angles[i] + angles[0] + 2 * math.pi) / 2
            ax = radius * 1.1 * math.cos(mid_angle)
            ay = radius * 1.1 * math.sin(mid_angle)
            
            # Arrow direction (tangent to circle)
            arrow_angle = mid_angle + math.pi / 2
            arrow_len = 0.15
            data.append(go.Scatter(
                x=[ax, ax + arrow_len * math.cos(arrow_angle)],
                y=[ay, ay + arrow_len * math.sin(arrow_angle)],
                mode='lines',
                line=dict(color=colors['ring'], width=2),
                hoverinfo='none',
                showlegend=False,
            ))
        
        # Connection lines from MSSs to MHs
        for mss_id in range(num_mss):
            for mh_idx in range(3):
                mx, my = mh_positions[mss_id][mh_idx]
                data.append(go.Scatter(
                    x=[mss_x[mss_id], mx],
                    y=[mss_y[mss_id], my],
                    mode='lines',
                    line=dict(color=colors['connection'], width=1, dash='dot'),
                    hoverinfo='none',
                    showlegend=False,
                ))
        
        # MSS nodes
        mss_color_list = [mss_colors.get(i, colors['mss_normal']) for i in range(num_mss)]
        mss_sizes = [50 if mss_colors.get(i) else 45 for i in range(num_mss)]
        
        data.append(go.Scatter(
            x=mss_x,
            y=mss_y,
            mode='markers+text',
            marker=dict(
                size=mss_sizes,
                color=mss_color_list,
                line=dict(width=3, color='white'),
                symbol='square'
            ),
            text=[f'MSS_{i}' for i in range(num_mss)],
            textposition='top center',
            textfont=dict(color='white', size=11, family='Arial Black'),
            hoverinfo='text',
            hovertext=[f'MSS_{i}<br>Status: {"Holding Token" if mss_colors.get(i) == colors["mss_holding"] else "Normal"}' 
                       for i in range(num_mss)],
            showlegend=False,
        ))
        
        # MH nodes
        for mss_id in range(num_mss):
            for mh_idx in range(3):
                mx, my = mh_positions[mss_id][mh_idx]
                mh_name = mh_names[mss_id][mh_idx]
                mh_color = mh_colors.get(mh_name, colors['mh_normal'])
                mh_size = 25 if mh_colors.get(mh_name) else 20
                
                data.append(go.Scatter(
                    x=[mx],
                    y=[my],
                    mode='markers+text',
                    marker=dict(
                        size=mh_size,
                        color=mh_color,
                        line=dict(width=2, color='white'),
                        symbol='circle'
                    ),
                    text=[mh_name.split('_')[-1]],  # Just show A, B, C
                    textposition='bottom center',
                    textfont=dict(color='white', size=9),
                    hoverinfo='text',
                    hovertext=f'{mh_name}<br>MSS_{mss_id}',
                    showlegend=False,
                ))
        
        # Request message (small square)
        if request_msg:
            data.append(go.Scatter(
                x=[request_msg['x']],
                y=[request_msg['y']],
                mode='markers',
                marker=dict(
                    size=12,
                    color=colors['request_msg'],
                    symbol='square',
                    line=dict(width=1, color='white')
                ),
                hoverinfo='text',
                hovertext='REQUEST',
                showlegend=False,
            ))
        
        # Release message (small square)
        if release_msg:
            data.append(go.Scatter(
                x=[release_msg['x']],
                y=[release_msg['y']],
                mode='markers',
                marker=dict(
                    size=12,
                    color=colors['release_msg'],
                    symbol='square',
                    line=dict(width=1, color='white')
                ),
                hoverinfo='text',
                hovertext='RELEASE',
                showlegend=False,
            ))
        
        # Handoff line (dashed arrow showing MH movement)
        if handoff_line:
            data.append(go.Scatter(
                x=[handoff_line['x1'], handoff_line['x2']],
                y=[handoff_line['y1'], handoff_line['y2']],
                mode='lines',
                line=dict(color=colors['mh_handoff'], width=3, dash='dash'),
                hoverinfo='none',
                showlegend=False,
            ))
        
        # Token
        token_angle = (token_pos / num_mss) * 2 * math.pi - math.pi / 2
        tx = radius * math.cos(token_angle)
        ty = radius * math.sin(token_angle)
        
        data.append(go.Scatter(
            x=[tx],
            y=[ty],
            mode='markers',
            marker=dict(
                size=30,
                color=token_color,
                symbol='circle',
                line=dict(width=3, color='#333')
            ),
            hoverinfo='text',
            hovertext=f'TOKEN<br>Position: {token_pos:.1f}',
            showlegend=False,
        ))
        
        # Log text box
        data.append(go.Scatter(
            x=[0],
            y=[-3.2],
            mode='text',
            text=[f'<b>{log_text}</b>'],
            textfont=dict(size=12, color='white'),
            hoverinfo='none',
            showlegend=False,
        ))
        
        return go.Frame(data=data, name=str(frame_num))
    
    frame_idx = 0
    
    # ═══════════════════════════════════════════════════════════════
    # PHASE 1: Free token circulating (a few rotations)
    # ═══════════════════════════════════════════════════════════════
    
    log_entries.append("Phase 1: Free token circulates on the ring...")
    
    for pos in range(req_mss_id * steps_between_mss):
        token_pos = pos / steps_between_mss
        frames.append(create_frame(
            token_pos=token_pos,
            token_color=colors['token_free'],
            mss_colors={},
            mh_colors={},
            frame_num=frame_idx,
            log_text="⚪ Free token circulating..."
        ))
        frame_idx += 1
    
    # ═══════════════════════════════════════════════════════════════
    # PHASE 2: MH sends REQUEST to its MSS
    # ═══════════════════════════════════════════════════════════════
    
    log_entries.append(f"Phase 2: {requesting_mh} sends REQUEST to MSS_{req_mss_id}")
    
    # Token continues while request is being sent
    mh_x, mh_y = mh_positions[req_mss_id][req_mh_idx]
    mss_target_x, mss_target_y = mss_x[req_mss_id], mss_y[req_mss_id]
    
    for step in range(message_steps):
        t = step / message_steps
        msg_x = interpolate(mh_x, mss_target_x, t)
        msg_y = interpolate(mh_y, mss_target_y, t)
        
        token_pos = req_mss_id + step / (steps_between_mss * 2)
        
        frames.append(create_frame(
            token_pos=token_pos,
            token_color=colors['token_free'],
            mss_colors={},
            mh_colors={requesting_mh: colors['mh_requesting']},
            request_msg={'x': msg_x, 'y': msg_y},
            frame_num=frame_idx,
            log_text=f"📤 {requesting_mh} sending REQUEST to MSS_{req_mss_id}"
        ))
        frame_idx += 1
    
    # ═══════════════════════════════════════════════════════════════
    # PHASE 3: MSS receives request, changes color, queues it
    # ═══════════════════════════════════════════════════════════════
    
    log_entries.append(f"Phase 3: MSS_{req_mss_id} receives request, adds to queue")
    
    for step in range(wait_steps // 2):
        token_pos = req_mss_id + 0.5 + step / steps_between_mss
        
        frames.append(create_frame(
            token_pos=token_pos,
            token_color=colors['token_free'],
            mss_colors={req_mss_id: colors['mss_pending']},
            mh_colors={requesting_mh: colors['mh_requesting']},
            frame_num=frame_idx,
            log_text=f"📋 MSS_{req_mss_id} queued request from {requesting_mh}"
        ))
        frame_idx += 1
    
    # ═══════════════════════════════════════════════════════════════
    # PHASE 4: Token continues around the ring back to requesting MSS
    # ═══════════════════════════════════════════════════════════════
    
    log_entries.append(f"Phase 4: Token continues circulating...")
    
    # Token travels from current position back around to req_mss_id
    current_token_pos = req_mss_id + 1
    target_token_pos = req_mss_id + num_mss  # Full circle
    
    for pos in range(int((target_token_pos - current_token_pos) * steps_between_mss)):
        token_pos = current_token_pos + pos / steps_between_mss
        
        frames.append(create_frame(
            token_pos=token_pos % num_mss,
            token_color=colors['token_free'],
            mss_colors={req_mss_id: colors['mss_pending']},
            mh_colors={requesting_mh: colors['mh_requesting']},
            frame_num=frame_idx,
            log_text=f"⚪ Token at MSS_{int(token_pos) % num_mss}, heading to MSS_{req_mss_id}"
        ))
        frame_idx += 1
    
    # ═══════════════════════════════════════════════════════════════
    # PHASE 5: Token arrives at MSS, MSS HOLDS it
    # ═══════════════════════════════════════════════════════════════
    
    log_entries.append(f"Phase 5: Token arrives at MSS_{req_mss_id}, MSS HOLDS token")
    
    for step in range(wait_steps):
        # Pulsing effect for holding
        pulse = 1 + 0.1 * math.sin(step * 0.5)
        
        frames.append(create_frame(
            token_pos=req_mss_id,
            token_color=colors['token_held'],
            mss_colors={req_mss_id: colors['mss_holding']},
            mh_colors={requesting_mh: colors['mh_requesting']},
            frame_num=frame_idx,
            log_text=f"🟢 MSS_{req_mss_id} HOLDING token, checking queue..."
        ))
        frame_idx += 1
    
    # ═══════════════════════════════════════════════════════════════
    # PHASE 6: MSS grants PERMISSION to MH
    # ═══════════════════════════════════════════════════════════════
    
    log_entries.append(f"Phase 6: MSS_{req_mss_id} grants PERMISSION to {requesting_mh}")
    
    # Permission message from MSS to MH
    for step in range(message_steps):
        t = step / message_steps
        msg_x = interpolate(mss_target_x, mh_x, t)
        msg_y = interpolate(mss_target_y, mh_y, t)
        
        frames.append(create_frame(
            token_pos=req_mss_id,
            token_color=colors['token_held'],
            mss_colors={req_mss_id: colors['mss_granting']},
            mh_colors={requesting_mh: colors['mh_requesting']},
            release_msg={'x': msg_x, 'y': msg_y},  # Using blue for permission
            frame_num=frame_idx,
            log_text=f"📨 MSS_{req_mss_id} sending PERMISSION to {requesting_mh}"
        ))
        frame_idx += 1
    
    # ═══════════════════════════════════════════════════════════════
    # PHASE 7: MH enters Critical Section
    # ═══════════════════════════════════════════════════════════════
    
    log_entries.append(f"Phase 7: {requesting_mh} enters CRITICAL SECTION")
    
    for step in range(wait_steps):
        frames.append(create_frame(
            token_pos=req_mss_id,
            token_color=colors['token_held'],
            mss_colors={req_mss_id: colors['mss_holding']},
            mh_colors={requesting_mh: colors['mh_in_cs']},
            frame_num=frame_idx,
            log_text=f"🟣 {requesting_mh} in CRITICAL SECTION..."
        ))
        frame_idx += 1
    
    # ═══════════════════════════════════════════════════════════════
    # PHASE 8: MH sends RELEASE to MSS
    # ═══════════════════════════════════════════════════════════════
    
    log_entries.append(f"Phase 8: {requesting_mh} sends RELEASE to MSS_{req_mss_id}")
    
    for step in range(message_steps):
        t = step / message_steps
        msg_x = interpolate(mh_x, mss_target_x, t)
        msg_y = interpolate(mh_y, mss_target_y, t)
        
        frames.append(create_frame(
            token_pos=req_mss_id,
            token_color=colors['token_held'],
            mss_colors={req_mss_id: colors['mss_holding']},
            mh_colors={requesting_mh: colors['mh_in_cs']},
            release_msg={'x': msg_x, 'y': msg_y},
            frame_num=frame_idx,
            log_text=f"📤 {requesting_mh} sending RELEASE to MSS_{req_mss_id}"
        ))
        frame_idx += 1
    
    # ═══════════════════════════════════════════════════════════════
    # PHASE 9: MSS releases token, another MH requests
    # ═══════════════════════════════════════════════════════════════
    
    log_entries.append(f"Phase 9: MSS_{req_mss_id} releases token; {handoff_mh} sends request")
    
    # Another MH (handoff_mh) sends a request just as token leaves
    handoff_mh_x, handoff_mh_y = mh_positions[handoff_mss_id][handoff_mh_idx]
    handoff_mss_target_x, handoff_mss_target_y = mss_x[handoff_mss_id], mss_y[handoff_mss_id]
    
    for step in range(message_steps):
        t = step / message_steps
        msg_x = interpolate(handoff_mh_x, handoff_mss_target_x, t)
        msg_y = interpolate(handoff_mh_y, handoff_mss_target_y, t)
        
        token_pos = req_mss_id + t * 0.5
        
        frames.append(create_frame(
            token_pos=token_pos,
            token_color=colors['token_free'],
            mss_colors={},
            mh_colors={handoff_mh: colors['mh_requesting']},
            request_msg={'x': msg_x, 'y': msg_y},
            frame_num=frame_idx,
            log_text=f"📤 {handoff_mh} sending REQUEST to MSS_{handoff_mss_id}"
        ))
        frame_idx += 1
    
    # MSS receives the request
    for step in range(wait_steps // 2):
        token_pos = req_mss_id + 0.5 + step / (steps_between_mss * 2)
        
        frames.append(create_frame(
            token_pos=token_pos,
            token_color=colors['token_free'],
            mss_colors={handoff_mss_id: colors['mss_pending']},
            mh_colors={handoff_mh: colors['mh_requesting']},
            frame_num=frame_idx,
            log_text=f"📋 MSS_{handoff_mss_id} queued request from {handoff_mh}"
        ))
        frame_idx += 1
    
    # ═══════════════════════════════════════════════════════════════
    # PHASE 10: HANDOFF - MH moves to different MSS!
    # ═══════════════════════════════════════════════════════════════
    
    log_entries.append(f"Phase 10: HANDOFF! {handoff_mh} moves from MSS_{handoff_mss_id} to MSS_{handoff_target_mss}")
    
    new_mh_x, new_mh_y = mh_positions[handoff_target_mss][0]  # Takes position A at new MSS
    
    for step in range(message_steps * 2):
        t = step / (message_steps * 2)
        current_mh_x = interpolate(handoff_mh_x, new_mh_x, t)
        current_mh_y = interpolate(handoff_mh_y, new_mh_y, t)
        
        token_pos = req_mss_id + 1 + step / (steps_between_mss * 2)
        
        frames.append(create_frame(
            token_pos=token_pos % num_mss,
            token_color=colors['token_free'],
            mss_colors={handoff_mss_id: colors['mss_pending']},
            mh_colors={handoff_mh: colors['mh_handoff']},
            handoff_line={'x1': handoff_mh_x, 'y1': handoff_mh_y, 'x2': current_mh_x, 'y2': current_mh_y},
            frame_num=frame_idx,
            log_text=f"📱 HANDOFF: {handoff_mh} moving to MSS_{handoff_target_mss}..."
        ))
        frame_idx += 1
    
    # ═══════════════════════════════════════════════════════════════
    # PHASE 11: Request KILLED at old MSS
    # ═══════════════════════════════════════════════════════════════
    
    log_entries.append(f"Phase 11: Request KILLED at MSS_{handoff_mss_id}!")
    
    for step in range(wait_steps):
        token_pos = (req_mss_id + 2 + step / steps_between_mss) % num_mss
        
        # Flash effect for killed request
        if step % 4 < 2:
            mss_col = '#FF0000'  # Red flash
        else:
            mss_col = colors['mss_normal']
        
        frames.append(create_frame(
            token_pos=token_pos,
            token_color=colors['token_free'],
            mss_colors={handoff_mss_id: mss_col},
            mh_colors={},
            frame_num=frame_idx,
            log_text=f"❌ Request from {handoff_mh} KILLED at MSS_{handoff_mss_id}!"
        ))
        frame_idx += 1
    
    # ═══════════════════════════════════════════════════════════════
    # PHASE 12: MH re-registers at new MSS
    # ═══════════════════════════════════════════════════════════════
    
    log_entries.append(f"Phase 12: {handoff_mh} re-registers request at MSS_{handoff_target_mss}")
    
    new_mss_target_x, new_mss_target_y = mss_x[handoff_target_mss], mss_y[handoff_target_mss]
    
    for step in range(message_steps):
        t = step / message_steps
        msg_x = interpolate(new_mh_x, new_mss_target_x, t)
        msg_y = interpolate(new_mh_y, new_mss_target_y, t)
        
        token_pos = (req_mss_id + 3 + step / (steps_between_mss * 2)) % num_mss
        
        frames.append(create_frame(
            token_pos=token_pos,
            token_color=colors['token_free'],
            mss_colors={},
            mh_colors={handoff_mh: colors['mh_requesting']},
            request_msg={'x': msg_x, 'y': msg_y},
            frame_num=frame_idx,
            log_text=f"📤 {handoff_mh} RE-REGISTERING at MSS_{handoff_target_mss}"
        ))
        frame_idx += 1
    
    # New MSS receives request
    for step in range(wait_steps // 2):
        token_pos = (req_mss_id + 3.5 + step / steps_between_mss) % num_mss
        
        frames.append(create_frame(
            token_pos=token_pos,
            token_color=colors['token_free'],
            mss_colors={handoff_target_mss: colors['mss_pending']},
            mh_colors={handoff_mh: colors['mh_requesting']},
            frame_num=frame_idx,
            log_text=f"📋 MSS_{handoff_target_mss} queued NEW request from {handoff_mh}"
        ))
        frame_idx += 1
    
    # ═══════════════════════════════════════════════════════════════
    # PHASE 13: Token reaches new MSS, grants to handoff MH
    # ═══════════════════════════════════════════════════════════════
    
    log_entries.append(f"Phase 13: Token reaches MSS_{handoff_target_mss}, grants to {handoff_mh}")
    
    # Token travels to new MSS
    current_pos = (req_mss_id + 4) % num_mss
    while current_pos != handoff_target_mss:
        for step in range(steps_between_mss):
            token_pos = current_pos + step / steps_between_mss
            
            frames.append(create_frame(
                token_pos=token_pos % num_mss,
                token_color=colors['token_free'],
                mss_colors={handoff_target_mss: colors['mss_pending']},
                mh_colors={handoff_mh: colors['mh_requesting']},
                frame_num=frame_idx,
                log_text=f"⚪ Token heading to MSS_{handoff_target_mss}..."
            ))
            frame_idx += 1
        current_pos = (current_pos + 1) % num_mss
    
    # Token at new MSS, holding
    for step in range(wait_steps):
        frames.append(create_frame(
            token_pos=handoff_target_mss,
            token_color=colors['token_held'],
            mss_colors={handoff_target_mss: colors['mss_holding']},
            mh_colors={handoff_mh: colors['mh_requesting']},
            frame_num=frame_idx,
            log_text=f"🟢 MSS_{handoff_target_mss} HOLDING token..."
        ))
        frame_idx += 1
    
    # Grant to MH
    for step in range(message_steps):
        t = step / message_steps
        msg_x = interpolate(new_mss_target_x, new_mh_x, t)
        msg_y = interpolate(new_mss_target_y, new_mh_y, t)
        
        frames.append(create_frame(
            token_pos=handoff_target_mss,
            token_color=colors['token_held'],
            mss_colors={handoff_target_mss: colors['mss_granting']},
            mh_colors={handoff_mh: colors['mh_requesting']},
            release_msg={'x': msg_x, 'y': msg_y},
            frame_num=frame_idx,
            log_text=f"📨 MSS_{handoff_target_mss} granting PERMISSION to {handoff_mh}"
        ))
        frame_idx += 1
    
    # MH in CS
    for step in range(wait_steps):
        frames.append(create_frame(
            token_pos=handoff_target_mss,
            token_color=colors['token_held'],
            mss_colors={handoff_target_mss: colors['mss_holding']},
            mh_colors={handoff_mh: colors['mh_in_cs']},
            frame_num=frame_idx,
            log_text=f"🟣 {handoff_mh} in CRITICAL SECTION..."
        ))
        frame_idx += 1
    
    # ═══════════════════════════════════════════════════════════════
    # PHASE 14: Return to normal - free token circulates
    # ═══════════════════════════════════════════════════════════════
    
    log_entries.append("Phase 14: Normal operation resumes, free token circulates")
    
    for step in range(steps_between_mss * 3):
        token_pos = (handoff_target_mss + step / steps_between_mss) % num_mss
        
        frames.append(create_frame(
            token_pos=token_pos,
            token_color=colors['token_free'],
            mss_colors={},
            mh_colors={},
            frame_num=frame_idx,
            log_text=f"⚪ Normal operation: free token circulating..."
        ))
        frame_idx += 1
    
    # Create the initial figure
    initial_frame = frames[0] if frames else None
    
    fig = go.Figure(
        data=initial_frame.data if initial_frame else [],
        frames=frames,
        layout=go.Layout(
            title=dict(
                text='<b>Token Ring Mutual Exclusion - Complete Lifecycle</b>',
                font=dict(color='white', size=16),
                x=0.5,
            ),
            xaxis=dict(
                showgrid=False, zeroline=False, showticklabels=False,
                range=[-3.5, 3.5],
            ),
            yaxis=dict(
                showgrid=False, zeroline=False, showticklabels=False,
                range=[-3.8, 3.5],
                scaleanchor='x',
            ),
            plot_bgcolor='#1a1a2e',
            paper_bgcolor='#1a1a2e',
            height=650,
            margin=dict(l=20, r=20, t=60, b=60),
            showlegend=False,
        )
    )
    
    return fig, log_entries


def create_static_ring_with_mhs(mss_list):
    """Create a static ring visualization with MHs for the DME simulation tabs"""
    n = len(mss_list)
    radius = 2.0
    mh_radius = 0.5
    angles = [2 * math.pi * i / n - math.pi / 2 for i in range(n)]
    
    xs = [radius * math.cos(a) for a in angles]
    ys = [radius * math.sin(a) for a in angles]
    
    fig = go.Figure()
    
    # Ring circle
    ring_angles = np.linspace(0, 2 * np.pi, 100)
    fig.add_trace(go.Scatter(
        x=[radius * 1.05 * math.cos(a) for a in ring_angles],
        y=[radius * 1.05 * math.sin(a) for a in ring_angles],
        mode='lines',
        line=dict(color='#555', width=2),
        hoverinfo='none',
        showlegend=False,
    ))
    
    # Arrows between MSS nodes
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
    sizes = [55 if mss.has_token else 45 for mss in mss_list]
    
    fig.add_trace(go.Scatter(
        x=xs, y=ys,
        mode='markers+text',
        text=[f'MSS_{mss.id}' for mss in mss_list],
        textposition='top center',
        textfont=dict(size=12, color='black'),
        marker=dict(size=sizes, color=colors, line=dict(width=2, color='black'), symbol='square'),
        hoverinfo='text',
        hovertext=[f'MSS_{mss.id}<br>MHs: {len(mss.mobile_hosts)}' for mss in mss_list],
        showlegend=False,
    ))
    
    # MH nodes (3 per MSS)
    for idx, mss in enumerate(mss_list):
        for j, mh in enumerate(mss.mobile_hosts):
            offset_angle = angles[idx] + math.pi + (j - 1) * 0.4
            mx = xs[idx] + mh_radius * math.cos(offset_angle)
            my = ys[idx] + mh_radius * math.sin(offset_angle)
            
            # Connection line
            fig.add_trace(go.Scatter(
                x=[xs[idx], mx], y=[ys[idx], my],
                mode='lines',
                line=dict(width=1, color='#aaa', dash='dot'),
                hoverinfo='none',
                showlegend=False,
            ))
            
            # MH marker
            fig.add_trace(go.Scatter(
                x=[mx], y=[my],
                mode='markers+text',
                text=[mh.id.split('_')[-1]],  # Just show A, B, C
                textposition='bottom center',
                textfont=dict(size=9, color='#333'),
                marker=dict(size=18, color='#4CAF50', line=dict(width=1, color='white')),
                hoverinfo='text',
                hovertext=f'{mh.id} | P={mh.base_priority} | MSS_{mss.id}',
                showlegend=False,
            ))
    
    fig.update_layout(
        title=dict(text='<b>MSS-MH Token Ring Architecture</b>', font=dict(size=16)),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-3.5, 3.5]),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-3.5, 3.5], scaleanchor='x'),
        height=550,
        plot_bgcolor='white',
        paper_bgcolor='white',
        margin=dict(l=20, r=20, t=50, b=20),
    )
    
    return fig


# ═══════════════════════════════════════════════════════════════
#                    SCENARIO BUILDER
# ═══════════════════════════════════════════════════════════════

def build_scenario_with_3mhs(num_mss=4):
    """Build scenario with 3 MHs per MSS"""
    ring = RingTopology(num_mss)
    mhs = []
    
    for mss_id in range(num_mss):
        for mh_idx in range(3):
            mh_name = f"MH_{mss_id}_{chr(65 + mh_idx)}"
            priority = 5 + mh_idx + mss_id  # Varied priorities
            mh = MobileHost(mh_name, ring.nodes[mss_id], base_priority=priority)
            ring.nodes[mss_id].add_mh(mh)
            mhs.append(mh)
    
    return ring, mhs


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
.legend-box{display:inline-block;padding:8px 12px;margin:4px;border-radius:5px;font-size:12px;}
</style>
<div class="header">🔐 Token-Ring Mutual Exclusion — Replication Scheme</div>
<p class="sub">MSS-MH Architecture &nbsp;|&nbsp; Request Broadcasting &nbsp;|&nbsp;
Priority-Based Granting &nbsp;|&nbsp; Handoff Handling</p>
""", unsafe_allow_html=True)

# Session state initialization
if 'ring' not in st.session_state:
    r, m = build_scenario_with_3mhs(4)
    st.session_state.ring = r
    st.session_state.mhs = m
    st.session_state.tm = TokenManager(r)
    st.session_state.step = 0
    st.session_state.reqs_made: List[Request] = []
    st.session_state.anim_frame = 0

ring: RingTopology = st.session_state.ring
mhs: List[MobileHost] = st.session_state.mhs
tm: TokenManager = st.session_state.tm

# Sidebar
with st.sidebar:
    st.header('⚙️ Controls')

    if st.button('🔄 Reset Everything', use_container_width=True):
        r, m = build_scenario_with_3mhs(4)
        st.session_state.ring = r
        st.session_state.mhs = m
        st.session_state.tm = TokenManager(r)
        st.session_state.step = 0
        st.session_state.reqs_made = []
        st.session_state.anim_frame = 0
        st.rerun()

    st.markdown('---')
    st.subheader('📤 Send Request')
    
    # Group MHs by MSS
    mh_by_mss = {}
    for mh in mhs:
        mss_id = mh.current_mss.id
        if mss_id not in mh_by_mss:
            mh_by_mss[mss_id] = []
        mh_by_mss[mss_id].append(mh)
    
    selected_mss = st.selectbox('Select MSS', range(ring.n), format_func=lambda i: f'MSS_{i}')
    mhs_at_mss = mh_by_mss.get(selected_mss, [])
    
    if mhs_at_mss:
        selected_mh = st.selectbox(
            'Select MH',
            range(len(mhs_at_mss)),
            format_func=lambda i: f'{mhs_at_mss[i].id} (P={mhs_at_mss[i].base_priority})'
        )
        
        if st.button('📤 Send Request', use_container_width=True):
            req = mhs_at_mss[selected_mh].request_cs()
            if req:
                st.session_state.reqs_made.append(req)
                st.success(f'{mhs_at_mss[selected_mh].id} requested CS')
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
        g_opts = [f'{r.mh_id}' for r in granted_list]
        g_sel = st.selectbox('Select granted request', range(len(granted_list)), format_func=lambda i: g_opts[i])
        if st.button('✅ Mark Completed', use_container_width=True):
            chosen = granted_list[g_sel]
            tm.complete(chosen)
            for mh in mhs:
                if mh.id == chosen.mh_id:
                    mh.exit_cs()
            st.success(f'{chosen.mh_id} completed CS')
    else:
        st.info('No granted requests')

    st.markdown('---')
    holder = ring.token_holder()
    st.metric('Steps', st.session_state.step)
    st.metric('Token At', f'MSS_{holder.id}' if holder else '—')
    st.metric('Circulations', tm.circulations)


# ═══════════════════════════════════════════════════════════════
#                           TABS
# ═══════════════════════════════════════════════════════════════

tab1, tab2, tab3, tab4 = st.tabs([
    '🎬 1. Animation Demo',
    '📡 2. Request Propagation',
    '📋 3. Request Logs',
    '📊 4. Queue States',
])

# ────────── TAB 1: ANIMATION DEMO ──────────
with tab1:
    st.markdown('### Token Ring Animation — Complete Lifecycle Demo')
    
    # Animation configuration
    st.markdown('#### 🎛️ Animation Configuration')
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        num_mss_anim = st.selectbox('Number of MSSs', [4, 5, 6], index=2)
    
    with col2:
        # Generate MH options based on num_mss
        mh_options = []
        for mss_id in range(num_mss_anim):
            for mh_idx in range(3):
                mh_options.append(f"MH_{mss_id}_{chr(65 + mh_idx)}")
        
        requesting_mh = st.selectbox('Requesting MH', mh_options, index=3)  # Default MH_1_A
    
    with col3:
        # Filter out the requesting MH for handoff selection
        handoff_options = [mh for mh in mh_options if mh != requesting_mh]
        handoff_mh = st.selectbox('Handoff MH', handoff_options, index=4)  # Default different MH
    
    if st.button('🎬 Generate Animation', use_container_width=True, type='primary'):
        with st.spinner('Generating animation frames...'):
            anim_fig, log_entries = create_comprehensive_animation(
                num_mss=num_mss_anim,
                requesting_mh=requesting_mh,
                handoff_mh=handoff_mh
            )
            st.session_state.anim_fig = anim_fig
            st.session_state.log_entries = log_entries
    
    st.markdown('---')
    
    # Display animation
    if 'anim_fig' in st.session_state:
        st.markdown('#### 📺 Animation Viewer')
        
        st.info('''
        **🎮 Controls:** Use the slider below to step through the animation frame by frame.
        ''')
        
        anim_fig = st.session_state.anim_fig
        
        # Frame slider control
        if anim_fig.frames:
            frame_idx = st.slider(
                'Animation Frame',
                0,
                len(anim_fig.frames) - 1,
                0,
                key='frame_slider'
            )
            
            # Auto-play toggle
            col1, col2, col3 = st.columns([1, 1, 2])
            with col1:
                auto_play = st.checkbox('Auto-play')
            with col2:
                speed = st.selectbox('Speed', ['Slow', 'Medium', 'Fast'], index=1)
            
            if auto_play:
                speed_map = {'Slow': 0.3, 'Medium': 0.1, 'Fast': 0.05}
                time.sleep(speed_map[speed])
                if frame_idx < len(anim_fig.frames) - 1:
                    st.session_state.frame_slider = frame_idx + 1
                    st.rerun()
            
            # Display the current frame
            current_frame = anim_fig.frames[frame_idx]
            
            display_fig = go.Figure(
                data=current_frame.data,
                layout=anim_fig.layout
            )
            
            st.plotly_chart(display_fig, use_container_width=True)
    else:
        st.warning('👆 Configure and click "Generate Animation" to see the demo')
    
    # Legend
    st.markdown('---')
    st.markdown('#### 🎨 Color Legend')
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown('''
        **Token States:**
        - ⚪ White = Free token
        - 🟢 Green = Token held
        ''')
    
    with col2:
        st.markdown('''
        **MSS States:**
        - 🔵 Cyan = Normal
        - 🟡 Yellow = Has pending request
        - 🟢 Green = Holding token
        - 🟣 Purple = Granting permission
        ''')
    
    with col3:
        st.markdown('''
        **MH States:**
        - 🟢 Green = Normal
        - 🟠 Orange = Requesting
        - 🟣 Purple = In CS
        - 🟠 Orange (dashed) = Handoff
        ''')
    
    with col4:
        st.markdown('''
        **Messages:**
        - 🟧 Orange square = REQUEST
        - 🟦 Blue square = PERMISSION/RELEASE
        - ➖ Dashed line = Handoff path
        ''')
    
    # Phase descriptions
    st.markdown('---')
    st.markdown('#### 📜 Animation Phases')
    
    if 'log_entries' in st.session_state:
        for i, entry in enumerate(st.session_state.log_entries):
            st.markdown(f'**{i+1}.** {entry}')


# ────────── TAB 2: REQUEST PROPAGATION ──────────
with tab2:
    st.markdown('### Request Broadcasting & Priority-Based Granting')
    
    st.info('''
    **Replication mechanism:**
    1. MH sends REQUEST to its local MSS
    2. MSS broadcasts the request to ALL other MSSs
    3. Every MSS adds the request to its replicated global queue
    4. When token arrives, MSS grants to highest-priority LOCAL MH
    ''')
    
    # Static ring visualization
    static_fig = create_static_ring_with_mhs(ring.nodes)
    st.plotly_chart(static_fig, use_container_width=True)
    
    st.markdown('#### 📨 Broadcast Trace')
    if st.session_state.reqs_made:
        for req in st.session_state.reqs_made:
            other_mss = [f'MSS_{m.id}' for m in ring.nodes if m.id != req.source_mss_id]
            with st.expander(f'📨 {req.request_id} | {req.mh_id} → MSS_{req.source_mss_id} | P={req.priority} | {req.status}'):
                st.write(f'**Origin:** {req.mh_id} at MSS_{req.source_mss_id}')
                st.write(f'**Priority:** {req.priority} | **Lamport Time:** {req.timestamp}')
                st.write(f'**Broadcast to:** {", ".join(other_mss)}')
                st.write(f'**Status:** `{req.status}`')
    else:
        st.warning('No requests yet. Use the sidebar to send requests.')
    
    st.markdown('---')
    st.markdown('#### 📊 Global Priority Queue (Token Holder)')
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
    else:
        st.info('Queue is empty')

    st.markdown('---')
    st.markdown('#### 📝 Event Log')
    if tm.event_log:
        for ev in tm.event_log[-15:]:
            st.text(ev)
    else:
        st.info('No events yet')


# ────────── TAB 3: REQUEST LOGS ──────────
with tab3:
    st.markdown('### Replicated Request Logs at Each MSS')
    
    st.info('Every MSS keeps a complete replicated copy of all requests.')
    
    for mss in ring.nodes:
        token_badge = ' 🔑' if mss.has_token else ''
        mh_list = ', '.join([mh.id for mh in mss.mobile_hosts])
        
        with st.expander(f'📋 MSS_{mss.id}{token_badge} — MHs: [{mh_list}] — {len(mss.replicated_log)} entries', expanded=True):
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
                st.dataframe(pd.DataFrame(log_data), use_container_width=True, hide_index=True)
            else:
                st.caption('No entries yet')
    
    st.markdown('---')
    st.markdown('#### ✅ Replication Consistency Check')
    sets = []
    for mss in ring.nodes:
        ids = sorted(set(r.request_id for r in mss.replicated_log))
        sets.append(ids)
    
    if len(sets) > 1 and all(s == sets[0] for s in sets):
        st.success('✅ All MSSs have identical replicated request sets')
    elif not any(sets):
        st.info('No requests to compare yet')
    else:
        st.warning('⚠️ Logs not yet synchronized')


# ────────── TAB 4: QUEUE STATES ──────────
with tab4:
    st.markdown('### Queue State After Service')
    
    st.info('After a request is granted and completed, it is removed from the global queue.')
    
    cols = st.columns(ring.n)
    for idx_col, mss in enumerate(ring.nodes):
        with cols[idx_col]:
            tok = ' 🔑' if mss.has_token else ''
            st.markdown(f'#### MSS_{mss.id}{tok}')
            
            st.caption(f'**MHs:** {", ".join([mh.id for mh in mss.mobile_hosts])}')
            
            st.caption('**Local Queue:**')
            if mss.local_queue:
                ldf = pd.DataFrame([{
                    'MH': r.mh_id,
                    'P': r.priority,
                    'Status': r.status,
                } for r in mss.local_queue])
                st.dataframe(ldf, hide_index=True, use_container_width=True)
            else:
                st.success('Empty ✓')
            
            st.caption('**Global Queue:**')
            if mss.global_queue:
                gdf = pd.DataFrame([{
                    'MH': r.mh_id,
                    'P': r.priority,
                    'Status': r.status,
                } for i, r in enumerate(mss.global_queue[:5])])  # Show top 5
                st.dataframe(gdf, hide_index=True, use_container_width=True)
            else:
                st.success('Empty ✓')
    
    st.markdown('---')
    st.markdown('#### ✅ Completed Requests')
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
        st.warning('No completed requests yet')
    
    st.markdown('---')
    st.markdown('#### 📊 Per-MSS Statistics')
    st.dataframe(
        pd.DataFrame([mss.stats() for mss in ring.nodes]),
        use_container_width=True,
        hide_index=True,
    )


# Footer
st.markdown('---')
st.markdown('''
<div style='text-align:center;color:gray;font-size:.85rem;'>
<b>Token-Ring Mutual Exclusion with Replication</b><br>
MSS-MH Architecture · Lamport Clocks · Priority Queues · Request Broadcasting · Handoff Handling
</div>
''', unsafe_allow_html=True)
