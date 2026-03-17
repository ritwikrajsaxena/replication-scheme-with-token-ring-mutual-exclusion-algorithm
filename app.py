"""
Token-Ring Mutual Exclusion with Replication — GLOBAL PRIORITY FIXED
====================================================================
Tab 1: Original Token Ring Animation (full scenario with handoff)
Tab 2: Request Broadcasting Animation
Tab 3: Priority-Based Granting Animation (GLOBAL PRIORITY)
Tab 4: Replicated Request Logs
Tab 5: Queue States Animation
"""

import streamlit as st
import pandas as pd
import numpy as np
import math
import random
from dataclasses import dataclass, field
from typing import List, Optional, Dict
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
        result = []
        for s in range(self.n):
            for j in range(3):
                result.append((s, j, self.hn[s][j]))
        return result


def lerp(a, b, t):
    return a + (b - a) * max(0.0, min(1.0, t))


def generate_random_scenario(g: Geom, min_requests=1, max_requests=8):
    """Generate random requests with random priorities."""
    all_mhs = g.all_mhs()
    num_requests = random.randint(min_requests, min(max_requests, len(all_mhs)))
    selected = random.sample(all_mhs, num_requests)
    random.shuffle(selected)
    
    requests = []
    for timestamp, (mss_id, mh_idx, mh_name) in enumerate(selected, start=1):
        priority = random.randint(1, 10)
        req = Request(mh_id=mh_name, source_mss=mss_id, priority=priority, timestamp=timestamp)
        requests.append(req)
    
    return requests


# ═══════════════════════════════════════════════════════════════
#              TAB 1: ORIGINAL TOKEN RING ANIMATION
# ═══════════════════════════════════════════════════════════════

class MainTokenRingAnimator:
    """Full token ring animation with request, grant, and handoff."""
    
    C = dict(tf='#FFF', th='#0F0', mn='#00D4FF', mp='#FFD700', mh='#0F0',
             mg='#CC00FF', mr='#FF5722', mc='#9C27B0', ck='#87CEEB',
             ring='#444', conn='#666')

    def __init__(self, g: Geom, req_mh: str, ho_mh: str):
        self.g = g
        self.req_mh = req_mh
        self.ho_mh = ho_mh
        self.rms = int(req_mh.split('_')[1])
        self.rmi = ord(req_mh.split('_')[2]) - 65
        self.hms = int(ho_mh.split('_')[1])
        self.hmi = ord(ho_mh.split('_')[2]) - 65
        self.htgt = (self.hms + 1) % g.n
        self.frames = []
        self.logs = []
        self.tp = 0.0
        self.MS = 10
        self.ST = 8
        self.HL = 15
        self.MG = 8

    def _frame(self, tc, sc, hc, rm=None, pm=None, rl=None, hl=None, txt="", ovr=None):
        C, g = self.C, self.g
        data = []
        
        # Ring
        ra = np.linspace(0, 2 * np.pi, 100)
        data.append(go.Scatter(x=g.R*1.1*np.cos(ra), y=g.R*1.1*np.sin(ra),
                               mode='lines', line=dict(color=C['ring'], width=3), hoverinfo='none'))
        
        # Arrows
        for i in range(g.n):
            m = (g.ang[i] + g.ang[(i+1)%g.n]) / 2
            if i == g.n-1: m = (g.ang[i] + g.ang[0] + 2*math.pi) / 2
            ax, ay = g.R*1.1*math.cos(m), g.R*1.1*math.sin(m)
            data.append(go.Scatter(x=[ax, ax+.12*math.cos(m+math.pi/2)],
                                   y=[ay, ay+.12*math.sin(m+math.pi/2)],
                                   mode='lines', line=dict(color=C['ring'], width=2), hoverinfo='none'))
        
        # Connections
        lx, ly = [], []
        for s in range(g.n):
            for j in range(3):
                nm = g.hn[s][j]
                if ovr and nm in ovr: continue
                lx += [g.sx[s], g.hp[s][j][0], None]
                ly += [g.sy[s], g.hp[s][j][1], None]
        data.append(go.Scatter(x=lx, y=ly, mode='lines',
                               line=dict(color=C['conn'], width=1, dash='dot'), hoverinfo='none'))
        
        # MSS
        mc = [sc.get(i, C['mn']) for i in range(g.n)]
        ms = [55 if i in sc else 48 for i in range(g.n)]
        data.append(go.Scatter(x=g.sx, y=g.sy, mode='markers+text',
                               marker=dict(size=ms, color=mc, line=dict(width=3, color='white'), symbol='square'),
                               text=[f'MSS_{i}' for i in range(g.n)], textposition='top center',
                               textfont=dict(color='white', size=11), hoverinfo='none'))
        
        # MHs
        hx, hy, hcc, hs, ht = [], [], [], [], []
        for s in range(g.n):
            for j in range(3):
                nm = g.hn[s][j]
                mx, my = ovr[nm] if (ovr and nm in ovr) else g.hp[s][j]
                hx.append(mx); hy.append(my); ht.append(nm.split('_')[-1])
                hcc.append(hc.get(nm, '#4CAF50')); hs.append(28 if nm in hc else 22)
        data.append(go.Scatter(x=hx, y=hy, mode='markers+text',
                               marker=dict(size=hs, color=hcc, line=dict(width=2, color='white'), symbol='circle'),
                               text=ht, textposition='bottom center', textfont=dict(color='white', size=10), hoverinfo='none'))
        
        # Messages
        def _msg(obj, lbl, col):
            if obj:
                data.append(go.Scatter(x=[obj['x']], y=[obj['y']], opacity=1, mode='markers+text',
                                       text=[lbl], textposition='top center', textfont=dict(color='white', size=8),
                                       marker=dict(size=15, color=col, symbol='square', line=dict(width=2, color='white')), hoverinfo='none'))
            else:
                data.append(go.Scatter(x=[0], y=[0], opacity=0, hoverinfo='none'))
        _msg(rm, 'REQ', '#FF5722'); _msg(pm, 'PERM', '#9C27B0'); _msg(rl, 'REL', '#2196F3')
        
        # Handoff
        if hl:
            data.append(go.Scatter(x=[hl['x1'], hl['x2']], y=[hl['y1'], hl['y2']],
                                   mode='lines', line=dict(color='#FF9800', width=3, dash='dash'), hoverinfo='none'))
        else:
            data.append(go.Scatter(x=[0], y=[0], opacity=0, hoverinfo='none'))
        
        # Token
        ta = ((self.tp % g.n)/g.n)*2*math.pi - math.pi/2
        data.append(go.Scatter(x=[g.R*math.cos(ta)], y=[g.R*math.sin(ta)], mode='markers+text',
                               marker=dict(size=35, color=tc, symbol='circle', line=dict(width=4, color='#333')),
                               text=['🔑'], textfont=dict(size=14), hoverinfo='none'))
        
        # Log
        data.append(go.Scatter(x=[0], y=[-3.3], mode='text',
                               text=[f'<b>{txt}</b>'], textfont=dict(size=13, color='white'), hoverinfo='none'))
        return data

    def _to(self, tgt, **kw):
        cur = self.tp % self.g.n
        diff = (tgt - cur) % self.g.n
        if diff < 0.01: return
        s, e = self.tp, self.tp + diff
        for i in range(self.MS):
            self.tp = lerp(s, e, (i+1)/self.MS)
            self.frames.append(self._frame(**kw))

    def _hold(self, n, **kw):
        for _ in range(n): self.frames.append(self._frame(**kw))

    def _msg(self, fp, tp, key, **kw):
        for i in range(self.MG):
            t = (i+1)/self.MG
            kw[key] = {'x': lerp(fp[0], tp[0], t), 'y': lerp(fp[1], tp[1], t)}
            self.frames.append(self._frame(**kw))

    def generate(self):
        C, g = self.C, self.g
        rx, ry = g.hp[self.rms][self.rmi]
        smx, smy = g.sx[self.rms], g.sy[self.rms]
        hx, hy = g.hp[self.hms][self.hmi]
        hmx, hmy = g.sx[self.hms], g.sy[self.hms]
        nx, ny = g.hp[self.htgt][0]
        nmx, nmy = g.sx[self.htgt], g.sy[self.htgt]

        def L(s): self.logs.append(s)

        L(f"═══ PHASE 1: {self.req_mh} sends REQUEST to MSS_{self.rms} ═══")
        L(f"  • {self.req_mh} initiates critical section request")
        L(f"  • Request travels from MH to local MSS_{self.rms}")
        self._msg((rx,ry),(smx,smy),'rm', tc=C['tf'], sc={}, hc={self.req_mh:C['mr']},
                  txt=f"📤 {self.req_mh} → REQUEST → MSS_{self.rms}")

        L(f"═══ PHASE 2: MSS_{self.rms} queues and BROADCASTS ═══")
        L(f"  • MSS_{self.rms} adds request to local queue")
        L(f"  • MSS_{self.rms} broadcasts to {g.n-1} other MSSs")
        self._hold(self.ST, tc=C['tf'], sc={self.rms:C['mp']}, hc={self.req_mh:C['mr']},
                   txt=f"📋 MSS_{self.rms} queued · Broadcasting to {g.n-1} MSSs")

        L(f"═══ PHASE 3: Token circulates through ring ═══")
        for node in range(1, self.rms+1):
            L(f"  • Token visits MSS_{node}" + (" ← PENDING REQUEST FOUND!" if node==self.rms else " (empty queue)"))
            self._to(node, tc=C['tf'], sc={self.rms:C['mp']}, hc={self.req_mh:C['mr']},
                     txt=f"⚪ Token → MSS_{node}")
            if node != self.rms:
                L(f"    → MSS_{node} checks queue: empty → pass token")
                self._hold(self.ST, tc=C['tf'], sc={node:C['ck'], self.rms:C['mp']},
                           hc={self.req_mh:C['mr']}, txt=f"🔍 MSS_{node}: queue empty → pass")

        L(f"═══ PHASE 4: MSS_{self.rms} HOLDS token ═══")
        L(f"  • Pending request from {self.req_mh} found")
        L(f"  • Token held while granting process begins")
        self._hold(self.HL, tc=C['th'], sc={self.rms:C['mh']}, hc={self.req_mh:C['mr']},
                   txt=f"🟢 MSS_{self.rms} HOLDS token!")

        L(f"═══ PHASE 5: PERMISSION granted to {self.req_mh} ═══")
        L(f"  • MSS_{self.rms} sends PERMISSION message")
        L(f"  • {self.req_mh} receives permission to enter CS")
        self._msg((smx,smy),(rx,ry),'pm', tc=C['th'], sc={self.rms:C['mg']}, hc={self.req_mh:C['mr']},
                  txt=f"📨 PERMISSION → {self.req_mh}")

        L(f"═══ PHASE 6: {self.req_mh} in CRITICAL SECTION ═══")
        L(f"  • {self.req_mh} executing critical section code")
        L(f"  • Token remains held at MSS_{self.rms}")
        self._hold(self.HL, tc=C['th'], sc={self.rms:C['mh']}, hc={self.req_mh:C['mc']},
                   txt=f"🟣 {self.req_mh} in CS")

        L(f"═══ PHASE 7: {self.req_mh} sends RELEASE ═══")
        L(f"  • {self.req_mh} finished critical section")
        L(f"  • RELEASE message sent to MSS_{self.rms}")
        self._msg((rx,ry),(smx,smy),'rl', tc=C['th'], sc={self.rms:C['mh']}, hc={self.req_mh:C['mc']},
                  txt=f"📤 RELEASE → MSS_{self.rms}")

        L(f"═══ PHASE 8: {self.ho_mh} sends new REQUEST ═══")
        L(f"  • {self.ho_mh} initiates request at MSS_{self.hms}")
        L(f"  • Request queued and broadcast begins")
        self._msg((hx,hy),(hmx,hmy),'rm', tc=C['tf'], sc={}, hc={self.ho_mh:C['mr']},
                  txt=f"📤 {self.ho_mh} → REQUEST → MSS_{self.hms}")
        self._hold(self.ST, tc=C['tf'], sc={self.hms:C['mp']}, hc={self.ho_mh:C['mr']},
                   txt=f"📋 MSS_{self.hms} queued · Broadcasting")

        L(f"═══ PHASE 9: HANDOFF - {self.ho_mh} moves to MSS_{self.htgt} ═══")
        L(f"  • {self.ho_mh} physically moves to new cell")
        L(f"  • Moving from MSS_{self.hms} → MSS_{self.htgt}")
        for i in range(self.MG*2):
            t = (i+1)/(self.MG*2)
            cx, cy = lerp(hx, nx, t), lerp(hy, ny, t)
            self.tp += 0.5/(self.MG*2)
            self.frames.append(self._frame(
                tc=C['tf'], sc={self.hms:C['mp']}, hc={self.ho_mh:'#FF9800'},
                hl={'x1':hx,'y1':hy,'x2':cx,'y2':cy},
                txt=f"📱 HANDOFF → MSS_{self.htgt}",
                ovr={self.ho_mh:(cx,cy)}))

        L(f"═══ PHASE 10: Request KILLED at MSS_{self.hms} ═══")
        L(f"  • {self.ho_mh} no longer at MSS_{self.hms}")
        L(f"  • Original request invalidated")
        for i in range(self.HL):
            self.frames.append(self._frame(
                tc=C['tf'], sc={self.hms:'#F00' if i%6<3 else C['mn']}, hc={},
                txt=f"❌ Request KILLED at MSS_{self.hms}",
                ovr={self.ho_mh:(nx,ny)}))

        L(f"═══ PHASE 11: {self.ho_mh} re-registers at MSS_{self.htgt} ═══")
        L(f"  • {self.ho_mh} sends new request to MSS_{self.htgt}")
        L(f"  • New broadcast initiated")
        self._msg((nx,ny),(nmx,nmy),'rm', tc=C['tf'], sc={}, hc={self.ho_mh:C['mr']},
                  txt=f"📤 RE-REGISTER at MSS_{self.htgt}", ovr={self.ho_mh:(nx,ny)})
        self._hold(self.ST, tc=C['tf'], sc={self.htgt:C['mp']}, hc={self.ho_mh:C['mr']},
                   txt=f"📋 MSS_{self.htgt} queued new request", ovr={self.ho_mh:(nx,ny)})

        L(f"═══ PHASE 12: Token continues circulation ═══")
        cid = int(self.tp)+1
        safety = 0
        while cid % g.n != self.htgt and safety < g.n+2:
            tgt = cid % g.n
            L(f"  • Token visits MSS_{tgt} (empty) → pass")
            self._to(tgt, tc=C['tf'], sc={self.htgt:C['mp']}, hc={self.ho_mh:C['mr']},
                     txt=f"⚪ Token → MSS_{tgt}", ovr={self.ho_mh:(nx,ny)})
            self._hold(self.ST, tc=C['tf'], sc={tgt:C['ck'], self.htgt:C['mp']},
                       hc={self.ho_mh:C['mr']}, txt=f"🔍 MSS_{tgt}: empty → pass",
                       ovr={self.ho_mh:(nx,ny)})
            cid += 1; safety += 1

        L(f"═══ PHASE 13: Token grants to {self.ho_mh} at MSS_{self.htgt} ═══")
        L(f"  • Token arrives at MSS_{self.htgt}")
        L(f"  • {self.ho_mh}'s request is granted")
        self._to(self.htgt, tc=C['tf'], sc={self.htgt:C['mp']}, hc={self.ho_mh:C['mr']},
                 txt=f"⚪ Token → MSS_{self.htgt}", ovr={self.ho_mh:(nx,ny)})
        self._hold(self.HL, tc=C['th'], sc={self.htgt:C['mh']}, hc={self.ho_mh:C['mr']},
                   txt=f"🟢 MSS_{self.htgt} HOLDS token!", ovr={self.ho_mh:(nx,ny)})
        self._msg((nmx,nmy),(nx,ny),'pm', tc=C['th'], sc={self.htgt:C['mg']}, hc={self.ho_mh:C['mr']},
                  txt=f"📨 PERMISSION → {self.ho_mh}", ovr={self.ho_mh:(nx,ny)})

        L(f"═══ PHASE 14: {self.ho_mh} in CRITICAL SECTION ═══")
        L(f"  • {self.ho_mh} executing CS at new location")
        L(f"  • Handoff scenario completed successfully")
        self._hold(self.HL, tc=C['th'], sc={self.htgt:C['mh']}, hc={self.ho_mh:C['mc']},
                   txt=f"🟣 {self.ho_mh} in CS (new MSS)", ovr={self.ho_mh:(nx,ny)})

        L(f"═══ PHASE 15: Normal operation resumes ═══")
        L(f"  • Token released for further circulation")
        for i in range(self.MS*2):
            self.tp += 1.0/self.MS
            self.frames.append(self._frame(tc=C['tf'], sc={}, hc={}, txt="⚪ Normal circulation…"))

        pf = [go.Frame(data=f, name=str(i)) for i, f in enumerate(self.frames)]
        lo = go.Layout(
            title=dict(text='<b>Token Ring Animation — Full Scenario</b>', font=dict(size=16, color='white'), x=0.5),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-3.5,3.5]),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-3.8,3.5], scaleanchor='x'),
            plot_bgcolor='#1a1a2e', paper_bgcolor='#1a1a2e', height=700,
            margin=dict(l=20,r=20,t=60,b=20), showlegend=False,
            updatemenus=[dict(type='buttons', showactive=False, x=.05, y=-.05,
                xanchor='left', yanchor='top', direction='right',
                buttons=[
                    dict(label='▶ Play', method='animate',
                         args=[None, dict(frame=dict(duration=50,redraw=True), fromcurrent=True, transition=dict(duration=0))]),
                    dict(label='⏸ Pause', method='animate',
                         args=[[None], dict(frame=dict(duration=0,redraw=False), mode='immediate')])])],
            sliders=[dict(active=0, yanchor='top', xanchor='left',
                currentvalue=dict(font=dict(size=12, color='white'), prefix='Frame: ', visible=True),
                transition=dict(duration=0), pad=dict(b=10,t=30), len=.8, x=.2, y=-.05,
                steps=[dict(args=[[str(i)], dict(frame=dict(duration=0,redraw=True), mode='immediate')],
                       label='', method='animate') for i in range(len(pf))])])
        return self.frames[0], pf, lo, self.logs


# ═══════════════════════════════════════════════════════════════
#                TAB 2: BROADCASTING ANIMATION
# ═══════════════════════════════════════════════════════════════

class BroadcastAnimator:
    """Animates random requests being broadcast to all MSSs."""
    
    def __init__(self, g: Geom, requests: List[Request]):
        self.g = g
        self.requests = requests
        self.frames = []
        self.logs = []
        self.MSG_STEPS = 6
        self.HOLD_STEPS = 4
        self.queue_counts: Dict[int, int] = {i: 0 for i in range(g.n)}
        self.mss_has_request: Dict[int, set] = {i: set() for i in range(g.n)}

    def _build_frame(self, highlight_mh=None, highlight_mss=None,
                     msg_pos=None, msg_label="REQ", arrows=None,
                     log_text="", all_green=False):
        g = self.g
        data = []
        
        ra = np.linspace(0, 2 * np.pi, 100)
        data.append(go.Scatter(x=g.R*1.08*np.cos(ra), y=g.R*1.08*np.sin(ra),
                               mode='lines', line=dict(color='#888', width=2), hoverinfo='none'))
        
        mss_colors, mss_sizes = [], []
        for i in range(g.n):
            if all_green:
                mss_colors.append('#00CC00'); mss_sizes.append(55)
            elif highlight_mss and i in highlight_mss:
                mss_colors.append(highlight_mss[i]); mss_sizes.append(55)
            else:
                mss_colors.append('#00D4FF'); mss_sizes.append(48)
        
        data.append(go.Scatter(x=g.sx, y=g.sy, mode='markers+text',
                               marker=dict(size=mss_sizes, color=mss_colors, line=dict(width=3, color='white'), symbol='square'),
                               text=[f'MSS_{i}' for i in range(g.n)], textposition='top center',
                               textfont=dict(size=11, color='#333', family='Arial Black'), hoverinfo='none'))
        
        mh_x, mh_y, mh_colors, mh_sizes, mh_text = [], [], [], [], []
        for s in range(g.n):
            for j in range(3):
                mx, my = g.hp[s][j]
                mh_name = g.hn[s][j]
                mh_x.append(mx); mh_y.append(my); mh_text.append(mh_name.split('_')[-1])
                if highlight_mh and mh_name == highlight_mh:
                    mh_colors.append('#FF5722'); mh_sizes.append(30)
                else:
                    mh_colors.append('#4CAF50'); mh_sizes.append(20)
        
        data.append(go.Scatter(x=mh_x, y=mh_y, mode='markers+text',
                               marker=dict(size=mh_sizes, color=mh_colors, line=dict(width=2, color='white')),
                               text=mh_text, textposition='bottom center', textfont=dict(size=9, color='#333'), hoverinfo='none'))
        
        conn_x, conn_y = [], []
        for s in range(g.n):
            for j in range(3):
                mx, my = g.hp[s][j]
                conn_x.extend([g.sx[s], mx, None]); conn_y.extend([g.sy[s], my, None])
        data.append(go.Scatter(x=conn_x, y=conn_y, mode='lines',
                               line=dict(color='#ccc', width=1, dash='dot'), hoverinfo='none'))
        
        if arrows:
            data.append(go.Scatter(x=arrows['x'], y=arrows['y'], mode='lines',
                                   line=dict(color='#FF5722', width=2), hoverinfo='none'))
        else:
            data.append(go.Scatter(x=[None], y=[None], mode='lines', hoverinfo='none'))
        
        if msg_pos:
            data.append(go.Scatter(x=[msg_pos[0]], y=[msg_pos[1]], mode='markers+text',
                                   marker=dict(size=20, color='#FF5722', symbol='diamond', line=dict(width=2, color='white')),
                                   text=[msg_label], textposition='top center',
                                   textfont=dict(size=10, color='#FF5722', family='Arial Black'), hoverinfo='none'))
        else:
            data.append(go.Scatter(x=[None], y=[None], mode='markers', opacity=0, hoverinfo='none'))
        
        badge_x, badge_y, badge_text = [], [], []
        for i in range(g.n):
            badge_x.append(g.sx[i]); badge_y.append(g.sy[i] - 0.4)
            cnt = self.queue_counts.get(i, 0)
            badge_text.append(f"Q:{cnt}" if cnt > 0 else "")
        data.append(go.Scatter(x=badge_x, y=badge_y, mode='text', text=badge_text,
                               textfont=dict(size=12, color='#FF5722', family='Arial Black'), hoverinfo='none'))
        
        data.append(go.Scatter(x=[0], y=[-3.1], mode='text', text=[f'<b>{log_text}</b>'],
                               textfont=dict(size=14, color='#333'), hoverinfo='none'))
        return data

    def build(self):
        g = self.g
        arrows_x, arrows_y = [], []
        
        self.logs.append("═══════════════════════════════════════════════════════")
        self.logs.append("          BROADCAST ANIMATION - DETAILED LOG           ")
        self.logs.append("═══════════════════════════════════════════════════════")
        
        for req_idx, req in enumerate(self.requests):
            src_mss = req.source_mss
            mh_name = req.mh_id
            mh_idx = ord(mh_name.split('_')[2]) - 65
            mh_x, mh_y = g.hp[src_mss][mh_idx]
            mss_x, mss_y = g.sx[src_mss], g.sy[src_mss]
            
            self.logs.append("")
            self.logs.append(f"━━━ REQUEST {req_idx + 1}/{len(self.requests)} ━━━")
            self.logs.append(f"  MH: {mh_name}")
            self.logs.append(f"  Priority: {req.priority}")
            self.logs.append(f"  Timestamp: {req.timestamp}")
            self.logs.append(f"  Source MSS: MSS_{src_mss}")
            self.logs.append("")
            
            self.logs.append(f"  [Step A] {mh_name} → MSS_{src_mss}")
            self.logs.append(f"           Sending request message...")
            for step in range(self.MSG_STEPS):
                t = (step + 1) / self.MSG_STEPS
                mx, my = lerp(mh_x, mss_x, t), lerp(mh_y, mss_y, t)
                self.frames.append(self._build_frame(
                    highlight_mh=mh_name, highlight_mss={src_mss: '#FFD700'},
                    msg_pos=(mx, my), msg_label=f"P{req.priority}",
                    arrows={'x': list(arrows_x), 'y': list(arrows_y)} if arrows_x else None,
                    log_text=f"📤 {mh_name} (P={req.priority}) → MSS_{src_mss}"))
            
            self.queue_counts[src_mss] += 1
            self.mss_has_request[src_mss].add(req.rid)
            
            self.logs.append(f"           ✓ MSS_{src_mss} received request")
            self.logs.append(f"           ✓ Added to local queue (Q={self.queue_counts[src_mss]})")
            
            for _ in range(self.HOLD_STEPS):
                self.frames.append(self._build_frame(
                    highlight_mh=mh_name, highlight_mss={src_mss: '#00CC00'},
                    arrows={'x': list(arrows_x), 'y': list(arrows_y)} if arrows_x else None,
                    log_text=f"📋 MSS_{src_mss} queued (P={req.priority}, T={req.timestamp})"))
            
            self.logs.append("")
            self.logs.append(f"  [Step B] BROADCASTING to {g.n - 1} other MSSs:")
            
            for tgt_mss in range(g.n):
                if tgt_mss == src_mss:
                    continue
                
                tgt_x, tgt_y = g.sx[tgt_mss], g.sy[tgt_mss]
                self.logs.append(f"           MSS_{src_mss} → MSS_{tgt_mss} ... ")
                
                for step in range(self.MSG_STEPS):
                    t = (step + 1) / self.MSG_STEPS
                    mx, my = lerp(mss_x, tgt_x, t), lerp(mss_y, tgt_y, t)
                    recv_mss = {i: '#87CEEB' for i in range(g.n) if i != src_mss and req.rid in self.mss_has_request[i]}
                    recv_mss[src_mss] = '#00CC00'
                    self.frames.append(self._build_frame(
                        highlight_mh=mh_name, highlight_mss=recv_mss,
                        msg_pos=(mx, my), msg_label=f"P{req.priority}",
                        arrows={'x': list(arrows_x), 'y': list(arrows_y)} if arrows_x else None,
                        log_text=f"📡 MSS_{src_mss} → MSS_{tgt_mss}..."))
                
                arrows_x.extend([mss_x, tgt_x, None])
                arrows_y.extend([mss_y, tgt_y, None])
                self.queue_counts[tgt_mss] += 1
                self.mss_has_request[tgt_mss].add(req.rid)
                
                self.logs.append(f"           ✓ MSS_{tgt_mss} received (Q={self.queue_counts[tgt_mss]})")
                
                recv_mss = {i: '#87CEEB' for i in range(g.n) if i != src_mss and req.rid in self.mss_has_request[i]}
                recv_mss[src_mss] = '#00CC00'
                recv_mss[tgt_mss] = '#00CC00'
                
                for _ in range(self.HOLD_STEPS):
                    self.frames.append(self._build_frame(
                        highlight_mh=mh_name, highlight_mss=recv_mss,
                        arrows={'x': list(arrows_x), 'y': list(arrows_y)},
                        log_text=f"✅ MSS_{tgt_mss} replicated!"))
            
            self.logs.append("")
            self.logs.append(f"  [DONE] Request from {mh_name} replicated to ALL {g.n} MSSs")
        
        self.logs.append("")
        self.logs.append("═══════════════════════════════════════════════════════")
        self.logs.append(f"  COMPLETE: {len(self.requests)} requests broadcast to all {g.n} MSSs")
        self.logs.append("═══════════════════════════════════════════════════════")
        
        for _ in range(self.HOLD_STEPS * 3):
            self.frames.append(self._build_frame(all_green=True, arrows={'x': arrows_x, 'y': arrows_y},
                               log_text=f"✅ All {len(self.requests)} requests replicated!"))
        
        plotly_frames = [go.Frame(data=f, name=str(i)) for i, f in enumerate(self.frames)]
        layout = go.Layout(
            title=dict(text='<b>Request Broadcasting Animation</b>', font=dict(size=16), x=0.5),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-3.3, 3.3]),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-3.5, 3.3], scaleanchor='x'),
            height=650, plot_bgcolor='#fafafa', paper_bgcolor='#fafafa',
            margin=dict(l=20, r=20, t=60, b=20), showlegend=False,
            updatemenus=[dict(type='buttons', showactive=False, x=0.05, y=-0.02,
                xanchor='left', yanchor='top', direction='right',
                buttons=[
                    dict(label='▶ Play', method='animate',
                         args=[None, dict(frame=dict(duration=40, redraw=True), fromcurrent=True, transition=dict(duration=0))]),
                    dict(label='⏸ Pause', method='animate',
                         args=[[None], dict(frame=dict(duration=0, redraw=False), mode='immediate')])])],
            sliders=[dict(active=0, yanchor='top', xanchor='left',
                currentvalue=dict(font=dict(size=11), prefix='Step: ', visible=True),
                transition=dict(duration=0), pad=dict(b=10, t=30), len=0.75, x=0.2, y=-0.02,
                steps=[dict(args=[[str(i)], dict(frame=dict(duration=0, redraw=True), mode='immediate')],
                       label='', method='animate') for i in range(len(plotly_frames))])])
        
        return self.frames[0], plotly_frames, layout, self.logs


# ═══════════════════════════════════════════════════════════════
#         TAB 3: PRIORITY GRANTING (GLOBAL PRIORITY FIXED)
# ═══════════════════════════════════════════════════════════════

class GrantingAnimator:
    """Animates token circulation with GLOBAL priority-based granting."""
    
    def __init__(self, g: Geom, requests: List[Request]):
        self.g = g
        self.requests = [Request(r.mh_id, r.source_mss, r.priority, r.timestamp) for r in requests]
        self.frames = []
        self.logs = []
        self.token_pos = 0.0
        self.MOVE_STEPS = 10
        self.HOLD_STEPS = 8
        self.GRANT_STEPS = 12

    def _get_global_highest_priority(self):
        """Find the globally highest priority pending request."""
        pending = [r for r in self.requests if r.status == "PENDING"]
        if not pending:
            return None
        return max(pending, key=lambda r: (r.priority, -r.timestamp))

    def _build_frame(self, token_color='#FFF', mss_colors=None, mh_colors=None, bar_data=None, log_text=""):
        g = self.g
        data = []
        mss_colors = mss_colors or {}
        mh_colors = mh_colors or {}
        
        ra = np.linspace(0, 2 * np.pi, 100)
        data.append(go.Scatter(x=g.R*1.08*np.cos(ra), y=g.R*1.08*np.sin(ra),
                               mode='lines', line=dict(color='#555', width=2), hoverinfo='none'))
        
        mss_cols = [mss_colors.get(i, '#00D4FF') for i in range(g.n)]
        mss_sizes = [55 if i in mss_colors else 48 for i in range(g.n)]
        data.append(go.Scatter(x=g.sx, y=g.sy, mode='markers+text',
                               marker=dict(size=mss_sizes, color=mss_cols, line=dict(width=3, color='white'), symbol='square'),
                               text=[f'MSS_{i}' for i in range(g.n)], textposition='top center',
                               textfont=dict(size=11, color='white', family='Arial Black'), hoverinfo='none'))
        
        mh_x, mh_y, mh_cols, mh_sizes = [], [], [], []
        for s in range(g.n):
            for j in range(3):
                mx, my = g.hp[s][j]
                mh_name = g.hn[s][j]
                mh_x.append(mx); mh_y.append(my)
                mh_cols.append(mh_colors.get(mh_name, '#4CAF50'))
                mh_sizes.append(28 if mh_name in mh_colors else 20)
        data.append(go.Scatter(x=mh_x, y=mh_y, mode='markers',
                               marker=dict(size=mh_sizes, color=mh_cols, line=dict(width=2, color='white')), hoverinfo='none'))
        
        token_angle = ((self.token_pos % g.n) / g.n) * 2 * math.pi - math.pi / 2
        tx, ty = g.R * math.cos(token_angle), g.R * math.sin(token_angle)
        data.append(go.Scatter(x=[tx], y=[ty], mode='markers+text',
                               marker=dict(size=38, color=token_color, symbol='circle', line=dict(width=4, color='#333')),
                               text=['🔑'], textfont=dict(size=16), hoverinfo='none'))
        
        if bar_data:
            bar_y = 2.5
            bar_x_start = 3.8
            bar_spacing = 0.35
            
            for i, (mh, pri, status, is_next) in enumerate(bar_data[:8]):
                bar_width = pri * 0.15
                if is_next: color = '#00CC00'
                elif status == 'GRANTED': color = '#9C27B0'
                elif status == 'COMPLETED': color = '#888'
                else: color = '#FF5722'
                
                y_pos = bar_y - i * bar_spacing
                data.append(go.Scatter(x=[bar_x_start, bar_x_start + bar_width], y=[y_pos, y_pos],
                                       mode='lines', line=dict(width=18, color=color), hoverinfo='none'))
                data.append(go.Scatter(x=[bar_x_start - 0.1], y=[y_pos], mode='text',
                                       text=[f"{mh.split('_')[-1]} P{pri}"], textposition='middle left',
                                       textfont=dict(size=9, color='white'), hoverinfo='none'))
        else:
            for _ in range(16):
                data.append(go.Scatter(x=[None], y=[None], mode='lines', hoverinfo='none'))
        
        data.append(go.Scatter(x=[0], y=[-3.2], mode='text', text=[f'<b>{log_text}</b>'],
                               textfont=dict(size=13, color='white'), hoverinfo='none'))
        return data

    def _get_bar_data(self, next_grant_req=None):
        """Get bar chart data sorted by priority, with next_grant_req highlighted."""
        all_reqs = sorted(self.requests, key=lambda r: (-r.priority, r.timestamp))
        return [(r.mh_id, r.priority, r.status, r is next_grant_req) for r in all_reqs]

    def build(self):
        g = self.g
        
        self.logs.append("═══════════════════════════════════════════════════════")
        self.logs.append("   PRIORITY-BASED GRANTING - GLOBAL PRIORITY FIXED    ")
        self.logs.append("═══════════════════════════════════════════════════════")
        self.logs.append("")
        self.logs.append("Initial pending requests (sorted by GLOBAL priority):")
        for r in sorted(self.requests, key=lambda x: (-x.priority, x.timestamp)):
            self.logs.append(f"  • {r.mh_id} at MSS_{r.source_mss}: Priority={r.priority}, T={r.timestamp}")
        self.logs.append("")
        self.logs.append("KEY PRINCIPLE: Token moves to MSS with HIGHEST GLOBAL PRIORITY request")
        self.logs.append("Token begins at MSS_0")
        self.logs.append("")
        
        granted_count = 0
        iteration = 0
        
        while True:
            # Find globally highest priority pending request
            next_req = self._get_global_highest_priority()
            if not next_req:
                break
            
            iteration += 1
            target_mss = next_req.source_mss
            
            self.logs.append(f"━━━ ITERATION {iteration} ━━━")
            self.logs.append(f"  GLOBAL SCAN: Checking all pending requests...")
            
            pending_by_priority = sorted([r for r in self.requests if r.status == "PENDING"],
                                        key=lambda r: (-r.priority, r.timestamp))
            for idx, r in enumerate(pending_by_priority[:3]):
                marker = "← HIGHEST (WINNER!)" if r is next_req else ""
                self.logs.append(f"    #{idx+1}: {r.mh_id} at MSS_{r.source_mss} P={r.priority} T={r.timestamp} {marker}")
            
            if len(pending_by_priority) > 3:
                self.logs.append(f"    ... and {len(pending_by_priority)-3} more")
            
            self.logs.append(f"  Decision: Token must go to MSS_{target_mss} (has {next_req.mh_id} P={next_req.priority})")
            self.logs.append("")
            
            # Calculate shortest path to target
            current_mss = int(self.token_pos) % g.n
            forward_dist = (target_mss - current_mss) % g.n
            
            if forward_dist == 0:
                self.logs.append(f"  Token already at MSS_{target_mss}!")
            else:
                self.logs.append(f"  Token path: MSS_{current_mss} → MSS_{target_mss} (distance: {forward_dist} hops)")
            
            # Move token to target MSS
            if forward_dist > 0:
                start_pos = self.token_pos
                end_pos = self.token_pos + forward_dist
                
                for step in range(self.MOVE_STEPS):
                    t = (step + 1) / self.MOVE_STEPS
                    self.token_pos = lerp(start_pos, end_pos, t)
                    bar_data = self._get_bar_data(next_req)
                    actual_mss = int(self.token_pos) % g.n
                    self.frames.append(self._build_frame(
                        token_color='#FFF',
                        mss_colors={target_mss: '#FFD700'} if step > self.MOVE_STEPS // 2 else {},
                        bar_data=bar_data,
                        log_text=f"⚪ Token moving to MSS_{target_mss} (global highest priority)..."))
            
            # Grant to this request
            self.logs.append(f"  ✓ Token arrives at MSS_{target_mss}")
            self.logs.append(f"  ✓ Granting to {next_req.mh_id} (GLOBAL highest priority={next_req.priority})")
            
            bar_data = self._get_bar_data(next_req)
            for _ in range(self.HOLD_STEPS):
                self.frames.append(self._build_frame(
                    token_color='#FFD700',
                    mss_colors={target_mss: '#FFD700'},
                    mh_colors={next_req.mh_id: '#FF5722'},
                    bar_data=bar_data,
                    log_text=f"🔍 MSS_{target_mss}: Checking request from {next_req.mh_id}"))
            
            for _ in range(self.GRANT_STEPS):
                self.frames.append(self._build_frame(
                    token_color='#00FF00',
                    mss_colors={target_mss: '#00FF00'},
                    mh_colors={next_req.mh_id: '#9C27B0'},
                    bar_data=bar_data,
                    log_text=f"🏆 GRANTED to {next_req.mh_id} (P={next_req.priority})"))
            
            next_req.status = "GRANTED"
            granted_count += 1
            
            bar_data = self._get_bar_data(None)
            for _ in range(self.GRANT_STEPS):
                self.frames.append(self._build_frame(
                    token_color='#00FF00',
                    mss_colors={target_mss: '#00FF00'},
                    mh_colors={next_req.mh_id: '#9C27B0'},
                    bar_data=bar_data,
                    log_text=f"🟣 {next_req.mh_id} in CRITICAL SECTION"))
            
            next_req.status = "COMPLETED"
            self.logs.append(f"  ✓ {next_req.mh_id} exits CS")
            
            bar_data = self._get_bar_data(None)
            for _ in range(self.HOLD_STEPS):
                self.frames.append(self._build_frame(
                    token_color='#FFF',
                    mss_colors={target_mss: '#00CC00'},
                    bar_data=bar_data,
                    log_text=f"✅ {next_req.mh_id} completed"))
            
            remaining = sum(1 for r in self.requests if r.status == "PENDING")
            self.logs.append(f"  Remaining: {remaining} pending request(s)")
            self.logs.append("")
        
        self.logs.append("═══════════════════════════════════════════════════════")
        self.logs.append(f"  COMPLETE: {granted_count}/{len(self.requests)} requests granted")
        self.logs.append(f"  SERVING ORDER (by GLOBAL priority):")
        
        completed = [r for r in self.requests if r.status == "COMPLETED"]
        for idx, r in enumerate(sorted(completed, key=lambda x: (-x.priority, x.timestamp)), 1):
            self.logs.append(f"    {idx}. {r.mh_id} at MSS_{r.source_mss} (P={r.priority}, T={r.timestamp})")
        
        self.logs.append("═══════════════════════════════════════════════════════")
        
        for _ in range(self.HOLD_STEPS * 2):
            self.frames.append(self._build_frame(
                token_color='#FFF',
                bar_data=self._get_bar_data(None),
                log_text=f"✅ All {len(self.requests)} requests served!"))
        
        plotly_frames = [go.Frame(data=f, name=str(i)) for i, f in enumerate(self.frames)]
        layout = go.Layout(
            title=dict(text='<b>Priority-Based Granting (GLOBAL PRIORITY)</b>',
                       font=dict(size=16, color='white'), x=0.5),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-3.5, 6]),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-3.6, 3.3], scaleanchor='x'),
            height=650, plot_bgcolor='#1a1a2e', paper_bgcolor='#1a1a2e',
            margin=dict(l=20, r=20, t=60, b=20), showlegend=False,
            updatemenus=[dict(type='buttons', showactive=False, x=0.05, y=-0.02,
                xanchor='left', yanchor='top', direction='right',
                buttons=[
                    dict(label='▶ Play', method='animate',
                         args=[None, dict(frame=dict(duration=50, redraw=True), fromcurrent=True, transition=dict(duration=0))]),
                    dict(label='⏸ Pause', method='animate',
                         args=[[None], dict(frame=dict(duration=0, redraw=False), mode='immediate')])])],
            sliders=[dict(active=0, yanchor='top', xanchor='left',
                currentvalue=dict(font=dict(size=11, color='white'), prefix='Step: ', visible=True),
                transition=dict(duration=0), pad=dict(b=10, t=30), len=0.7, x=0.2, y=-0.02,
                steps=[dict(args=[[str(i)], dict(frame=dict(duration=0, redraw=True), mode='immediate')],
                       label='', method='animate') for i in range(len(plotly_frames))])])
        
        return self.frames[0], plotly_frames, layout, self.logs, self.requests


# ═══════════════════════════════════════════════════════════════
#                  TAB 5: QUEUE STATES ANIMATION
# ═══════════════════════════════════════════════════════════════

class QueueAnimator:
    """Animates queue state transitions at each MSS (uses GLOBAL priority)."""
    
    def __init__(self, g: Geom, requests: List[Request]):
        self.g = g
        self.requests = [Request(r.mh_id, r.source_mss, r.priority, r.timestamp) for r in requests]
        self.frames = []
        self.logs = []
        self.token_pos = 0.0
        self.MOVE_STEPS = 8
        self.PROCESS_STEPS = 15

    def _get_global_highest_priority(self):
        pending = [r for r in self.requests if r.status == "PENDING"]
        if not pending:
            return None
        return max(pending, key=lambda r: (r.priority, -r.timestamp))

    def _build_frame(self, token_color='#FFF', mss_colors=None, queue_states=None, log_text=""):
        g = self.g
        data = []
        mss_colors = mss_colors or {}
        queue_states = queue_states or {}
        
        r_small = g.R * 0.85
        
        ra = np.linspace(0, 2 * np.pi, 100)
        data.append(go.Scatter(x=r_small*1.08*np.cos(ra), y=r_small*1.08*np.sin(ra),
                               mode='lines', line=dict(color='#555', width=2), hoverinfo='none'))
        
        sx = [r_small * math.cos(a) for a in g.ang]
        sy = [r_small * math.sin(a) for a in g.ang]
        mss_cols = [mss_colors.get(i, '#00D4FF') for i in range(g.n)]
        data.append(go.Scatter(x=sx, y=sy, mode='markers+text',
                               marker=dict(size=45, color=mss_cols, line=dict(width=2, color='white'), symbol='square'),
                               text=[f'MSS_{i}' for i in range(g.n)], textposition='middle center',
                               textfont=dict(size=10, color='white', family='Arial Black'), hoverinfo='none'))
        
        token_angle = ((self.token_pos % g.n) / g.n) * 2 * math.pi - math.pi / 2
        tx, ty = r_small * math.cos(token_angle), r_small * math.sin(token_angle)
        data.append(go.Scatter(x=[tx], y=[ty], mode='markers+text',
                               marker=dict(size=30, color=token_color, symbol='circle', line=dict(width=3, color='#333')),
                               text=['🔑'], textfont=dict(size=12), hoverinfo='none'))
        
        for mss_id in range(min(g.n, 6)):
            angle = g.ang[mss_id]
            qx = g.R * 1.6 * math.cos(angle)
            qy = g.R * 1.6 * math.sin(angle)
            
            qs = queue_states.get(mss_id, {'pending': [], 'granted': [], 'completed': []})
            p_count, g_count, c_count = len(qs['pending']), len(qs['granted']), len(qs['completed'])
            
            lines = [f"<b>MSS_{mss_id}</b>"]
            if p_count > 0:
                lines.append(f"⏳ Pend: {p_count}")
                for r in qs['pending'][:2]:
                    lines.append(f"  {r.mh_id.split('_')[-1]} P{r.priority}")
            if g_count > 0: lines.append(f"🟢 Grant: {g_count}")
            if c_count > 0: lines.append(f"✅ Done: {c_count}")
            if p_count == 0 and g_count == 0 and c_count == 0: lines.append("(empty)")
            
            data.append(go.Scatter(x=[qx], y=[qy], mode='text', text=["<br>".join(lines)],
                                   textfont=dict(size=10, color='white'), hoverinfo='none'))
        
        while len(data) < 10:
            data.append(go.Scatter(x=[None], y=[None], mode='markers', opacity=0, hoverinfo='none'))
        
        data.append(go.Scatter(x=[0], y=[-3.0], mode='text', text=[f'<b>{log_text}</b>'],
                               textfont=dict(size=13, color='white'), hoverinfo='none'))
        return data

    def _get_queue_states(self):
        states = {}
        for mss_id in range(self.g.n):
            states[mss_id] = {
                'pending': sorted([r for r in self.requests if r.source_mss == mss_id and r.status == "PENDING"],
                                 key=lambda r: (-r.priority, r.timestamp)),
                'granted': [r for r in self.requests if r.source_mss == mss_id and r.status == "GRANTED"],
                'completed': [r for r in self.requests if r.source_mss == mss_id and r.status == "COMPLETED"]
            }
        return states

    def build(self):
        g = self.g
        
        self.logs.append("═══════════════════════════════════════════════════════")
        self.logs.append("      QUEUE STATES - GLOBAL PRIORITY ORDERING          ")
        self.logs.append("═══════════════════════════════════════════════════════")
        self.logs.append("")
        self.logs.append("INITIAL QUEUE STATES:")
        qs = self._get_queue_states()
        for mss_id in range(g.n):
            p = qs[mss_id]['pending']
            if p:
                self.logs.append(f"  MSS_{mss_id}: {len(p)} pending")
                for r in p:
                    self.logs.append(f"    • {r.mh_id} P={r.priority} T={r.timestamp}")
            else:
                self.logs.append(f"  MSS_{mss_id}: (empty)")
        self.logs.append("")
        
        for _ in range(self.PROCESS_STEPS):
            self.frames.append(self._build_frame(queue_states=qs,
                               log_text=f"📋 Initial: {len(self.requests)} pending"))
        
        iteration = 0
        while True:
            next_req = self._get_global_highest_priority()
            if not next_req:
                break
            
            iteration += 1
            target_mss = next_req.source_mss
            
            self.logs.append(f"━━━ ITERATION {iteration}: Token → MSS_{target_mss} ━━━")
            self.logs.append(f"  Global highest: {next_req.mh_id} (P={next_req.priority})")
            
            current_mss = int(self.token_pos) % g.n
            forward_dist = (target_mss - current_mss) % g.n
            
            if forward_dist > 0:
                start_pos = self.token_pos
                for step in range(self.MOVE_STEPS):
                    t = (step + 1) / self.MOVE_STEPS
                    self.token_pos = lerp(start_pos, start_pos + forward_dist, t)
                    qs = self._get_queue_states()
                    self.frames.append(self._build_frame(
                        mss_colors={target_mss: '#87CEEB'},
                        queue_states=qs,
                        log_text=f"⚪ Token → MSS_{target_mss}"))
            
            qs = self._get_queue_states()
            self.logs.append(f"  Queue before: {len(qs[target_mss]['pending'])} pending")
            
            next_req.status = "GRANTED"
            qs = self._get_queue_states()
            for _ in range(self.PROCESS_STEPS):
                self.frames.append(self._build_frame(
                    token_color='#00FF00',
                    mss_colors={target_mss: '#00FF00'},
                    queue_states=qs,
                    log_text=f"🟢 Granted: {next_req.mh_id}"))
            
            next_req.status = "COMPLETED"
            qs = self._get_queue_states()
            self.logs.append(f"  Queue after: {len(qs[target_mss]['pending'])} pending, {len(qs[target_mss]['completed'])} done")
            
            for _ in range(self.PROCESS_STEPS):
                self.frames.append(self._build_frame(
                    token_color='#FFF',
                    mss_colors={target_mss: '#888'},
                    queue_states=qs,
                    log_text=f"✅ {next_req.mh_id} completed"))
            
            self.logs.append("")
        
        self.logs.append("═══════════════════════════════════════════════════════")
        self.logs.append("FINAL QUEUE STATES:")
        qs = self._get_queue_states()
        for mss_id in range(g.n):
            c = qs[mss_id]['completed']
            self.logs.append(f"  MSS_{mss_id}: {len(c)} completed")
        self.logs.append("═══════════════════════════════════════════════════════")
        
        for _ in range(self.PROCESS_STEPS * 2):
            self.frames.append(self._build_frame(
                queue_states=qs,
                log_text=f"✅ All queues cleared!"))
        
        plotly_frames = [go.Frame(data=f, name=str(i)) for i, f in enumerate(self.frames)]
        layout = go.Layout(
            title=dict(text='<b>Queue States (GLOBAL Priority)</b>',
                       font=dict(size=16, color='white'), x=0.5),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-4.5, 4.5]),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-3.5, 3.5], scaleanchor='x'),
            height=650, plot_bgcolor='#1a1a2e', paper_bgcolor='#1a1a2e',
            margin=dict(l=20, r=20, t=60, b=20), showlegend=False,
            updatemenus=[dict(type='buttons', showactive=False, x=0.05, y=-0.02,
                xanchor='left', yanchor='top', direction='right',
                buttons=[
                    dict(label='▶ Play', method='animate',
                         args=[None, dict(frame=dict(duration=50, redraw=True), fromcurrent=True, transition=dict(duration=0))]),
                    dict(label='⏸ Pause', method='animate',
                         args=[[None], dict(frame=dict(duration=0, redraw=False), mode='immediate')])])],
            sliders=[dict(active=0, yanchor='top', xanchor='left',
                currentvalue=dict(font=dict(size=11, color='white'), prefix='Step: ', visible=True),
                transition=dict(duration=0), pad=dict(b=10, t=30), len=0.7, x=0.2, y=-0.02,
                steps=[dict(args=[[str(i)], dict(frame=dict(duration=0, redraw=True), mode='immediate')],
                       label='', method='animate') for i in range(len(plotly_frames))])])
        
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
.log-box{background:#1e1e1e;color:#00ff00;font-family:monospace;font-size:12px;
padding:1rem;border-radius:8px;max-height:400px;overflow-y:auto;white-space:pre-wrap}
</style>
<div class="hdr">🔐 Token-Ring Mutual Exclusion — GLOBAL PRIORITY</div>
""", unsafe_allow_html=True)

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    '🔄 1. Token Ring',
    '📡 2. Broadcasting',
    '🏆 3. Priority Granting',
    '📋 4. Request Logs',
    '📊 5. Queue States'
])

with tab1:
    st.markdown('### 🔄 Token Ring Animation — Full Scenario')
    st.info('''
    **Complete demo showing:**
    1. MH sends request to local MSS
    2. MSS broadcasts to all other MSSs
    3. Token circulates, checking queues
    4. Token held when pending request found
    5. Permission granted, MH enters critical section
    6. **Handoff scenario:** MH moves to new cell mid-request
    ''')
    
    col1, col2, col3 = st.columns(3)
    num_mss = col1.selectbox('Number of MSSs', [4, 5, 6], index=2, key='t1_n')
    
    mh_opts = [f"MH_{i}_{chr(65+j)}" for i in range(num_mss) for j in range(3)]
    req_mh = col2.selectbox('Requesting MH', mh_opts, index=min(6, len(mh_opts)-1), key='t1_req')
    ho_opts = [m for m in mh_opts if m != req_mh]
    ho_mh = col3.selectbox('Handoff MH', ho_opts, index=min(4, len(ho_opts)-1), key='t1_ho')
    
    if st.button('🎬 Generate Animation', key='gen_t1', type='primary', use_container_width=True):
        with st.spinner('Building animation...'):
            g = Geom(num_mss)
            anim = MainTokenRingAnimator(g, req_mh, ho_mh)
            d0, frames, layout, logs = anim.generate()
            st.session_state.t1_anim = (d0, frames, layout, logs)
    
    if 't1_anim' in st.session_state:
        d0, frames, layout, logs = st.session_state.t1_anim
        fig = go.Figure(data=d0, frames=frames, layout=layout)
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown('#### 📖 Detailed Phase Log')
        log_text = "\n".join(logs)
        st.markdown(f'<div class="log-box">{log_text}</div>', unsafe_allow_html=True)

with tab2:
    st.markdown('### 📡 Request Broadcasting Animation')
    st.info('''
    **Random requests are generated and broadcast:**
    1. Each MH sends request to its local MSS
    2. Local MSS broadcasts to ALL other MSSs
    3. Every MSS ends up with complete replicated log
    ''')
    
    col1, col2, col3 = st.columns(3)
    num_mss_b = col1.selectbox('Number of MSSs', [4, 5, 6], index=1, key='t2_n')
    min_req = col2.number_input('Min requests', min_value=1, max_value=10, value=2, key='t2_min')
    max_req = col3.number_input('Max requests', min_value=int(min_req), max_value=15, value=max(int(min_req), 5), key='t2_max')
    
    if st.button('🎲 Generate Random Scenario', key='gen_t2', type='primary', use_container_width=True):
        with st.spinner('Generating random requests...'):
            g = Geom(num_mss_b)
            requests = generate_random_scenario(g, int(min_req), int(max_req))
            st.session_state.t2_requests = requests
            
            anim = BroadcastAnimator(g, requests)
            d0, frames, layout, logs = anim.build()
            st.session_state.t2_anim = (d0, frames, layout, logs)
    
    if 't2_anim' in st.session_state:
        st.markdown('#### 🎲 Generated Requests')
        req_df = pd.DataFrame([r.row() for r in st.session_state.t2_requests])
        st.dataframe(req_df, use_container_width=True, hide_index=True)
        
        st.markdown('#### 📺 Animation')
        d0, frames, layout, logs = st.session_state.t2_anim
        fig = go.Figure(data=d0, frames=frames, layout=layout)
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown('#### 📖 Detailed Broadcast Log')
        log_text = "\n".join(logs)
        st.markdown(f'<div class="log-box">{log_text}</div>', unsafe_allow_html=True)

with tab3:
    st.markdown('### 🏆 Priority-Based Granting Animation (GLOBAL PRIORITY FIXED)')
    st.success('''
    **✅ GLOBAL PRIORITY IMPLEMENTATION:**
    1. System scans ALL pending requests across ALL MSSs
    2. Token moves to MSS with **GLOBALLY HIGHEST PRIORITY** request
    3. Grants that request, then repeats
    4. Result: Requests served in strict priority order (10→5→2→1)
    ''')
    
    col1, col2, col3 = st.columns(3)
    num_mss_g = col1.selectbox('Number of MSSs', [4, 5, 6], index=1, key='t3_n')
    min_req_g = col2.number_input('Min requests', min_value=1, max_value=10, value=3, key='t3_min')
    max_req_g = col3.number_input('Max requests', min_value=int(min_req_g), max_value=15, value=max(int(min_req_g), 6), key='t3_max')
    
    if st.button('🎲 Generate Random Scenario', key='gen_t3', type='primary', use_container_width=True):
        with st.spinner('Generating random requests...'):
            g = Geom(num_mss_g)
            requests = generate_random_scenario(g, int(min_req_g), int(max_req_g))
            st.session_state.t3_requests = requests
            
            anim = GrantingAnimator(g, requests)
            d0, frames, layout, logs, final = anim.build()
            st.session_state.t3_anim = (d0, frames, layout, logs, final)
    
    if 't3_anim' in st.session_state:
        st.markdown('#### 🎲 Generated Requests (sorted by GLOBAL priority)')
        reqs_sorted = sorted(st.session_state.t3_requests, key=lambda r: (-r.priority, r.timestamp))
        req_df = pd.DataFrame([r.row() for r in reqs_sorted])
        st.dataframe(req_df, use_container_width=True, hide_index=True)
        
        st.markdown('#### 📺 Animation')
        d0, frames, layout, logs, final = st.session_state.t3_anim
        fig = go.Figure(data=d0, frames=frames, layout=layout)
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown('#### 📖 Detailed Granting Log')
        log_text = "\n".join(logs)
        st.markdown(f'<div class="log-box">{log_text}</div>', unsafe_allow_html=True)

with tab4:
    st.markdown('### 📋 Replicated Request Logs at Each MSS')
    st.info('''
    **Shows how logs are replicated:**
    - Each MSS stores ALL requests (local + broadcast received)
    - After broadcasting, every MSS has identical log
    ''')
    
    col1, col2, col3 = st.columns(3)
    num_mss_l = col1.selectbox('Number of MSSs', [4, 5, 6], index=1, key='t4_n')
    min_req_l = col2.number_input('Min requests', min_value=1, max_value=10, value=3, key='t4_min')
    max_req_l = col3.number_input('Max requests', min_value=int(min_req_l), max_value=15, value=max(int(min_req_l), 6), key='t4_max')
    
    if st.button('🎲 Generate Random Scenario', key='gen_t4', type='primary', use_container_width=True):
        with st.spinner('Generating...'):
            g = Geom(num_mss_l)
            requests = generate_random_scenario(g, int(min_req_l), int(max_req_l))
            st.session_state.t4_requests = requests
            st.session_state.t4_geom = g
    
    if 't4_requests' in st.session_state:
        requests = st.session_state.t4_requests
        g = st.session_state.t4_geom
        
        st.markdown('#### 🎲 Generated Requests')
        req_df = pd.DataFrame([r.row() for r in requests])
        st.dataframe(req_df, use_container_width=True, hide_index=True)
        
        st.markdown('---')
        st.markdown('#### 📋 Replicated Logs at Each MSS')
        
        cols = st.columns(g.n)
        for mss_id in range(g.n):
            with cols[mss_id]:
                st.markdown(f'**MSS_{mss_id}**')
                local = [r for r in requests if r.source_mss == mss_id]
                remote = [r for r in requests if r.source_mss != mss_id]
                
                st.success(f'📍 Local: {len(local)}')
                for r in local:
                    st.caption(f"• {r.mh_id} P={r.priority} T={r.timestamp}")
                
                st.info(f'📡 Replicated: {len(remote)}')
                for r in remote:
                    st.caption(f"• {r.mh_id} P={r.priority} T={r.timestamp}")
                
                st.metric("Total", len(requests))
        
        st.markdown('---')
        st.markdown('#### 📊 Summary')
        summary = []
        for mss_id in range(g.n):
            local = sum(1 for r in requests if r.source_mss == mss_id)
            summary.append({
                "MSS": f"MSS_{mss_id}",
                "Local": local,
                "Replicated": len(requests) - local,
                "Total": len(requests),
                "Msgs Sent": (g.n - 1) * local,
                "Msgs Recv": len(requests) - local
            })
        st.dataframe(pd.DataFrame(summary), use_container_width=True, hide_index=True)

with tab5:
    st.markdown('### 📊 Queue States Animation (GLOBAL PRIORITY)')
    st.info('''
    **Visualize queue transitions with GLOBAL priority:**
    - Token moves to MSS with highest priority request
    - Watch queues drain in priority order
    ''')
    
    col1, col2, col3 = st.columns(3)
    num_mss_q = col1.selectbox('Number of MSSs', [4, 5, 6], index=1, key='t5_n')
    min_req_q = col2.number_input('Min requests', min_value=1, max_value=10, value=3, key='t5_min')
    max_req_q = col3.number_input('Max requests', min_value=int(min_req_q), max_value=15, value=max(int(min_req_q), 5), key='t5_max')
    
    if st.button('🎲 Generate Random Scenario', key='gen_t5', type='primary', use_container_width=True):
        with st.spinner('Generating...'):
            g = Geom(num_mss_q)
            requests = generate_random_scenario(g, int(min_req_q), int(max_req_q))
            st.session_state.t5_requests = requests
            
            anim = QueueAnimator(g, requests)
            d0, frames, layout, logs, final = anim.build()
            st.session_state.t5_anim = (d0, frames, layout, logs, final)
    
    if 't5_anim' in st.session_state:
        st.markdown('#### 🎲 Generated Requests')
        req_df = pd.DataFrame([r.row() for r in st.session_state.t5_requests])
        st.dataframe(req_df, use_container_width=True, hide_index=True)
        
        st.markdown('#### 📺 Animation')
        d0, frames, layout, logs, final = st.session_state.t5_anim
        fig = go.Figure(data=d0, frames=frames, layout=layout)
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown('#### 📖 Detailed Queue Log')
        log_text = "\n".join(logs)
        st.markdown(f'<div class="log-box">{log_text}</div>', unsafe_allow_html=True)

st.markdown('---')
st.caption('Token-Ring Mutual Exclusion with GLOBAL PRIORITY | All requests served in strict priority order')
