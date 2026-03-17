"""
Token-Ring Mutual Exclusion with Replication — Complete 5-Tab App
=================================================================
Tab 1: Native JS Token Ring Animation (zero flicker)
Tab 2: Request Broadcasting visualization (MH → MSS → ALL MSSs)
Tab 3: Priority-Based Granting (bar charts + step-through)
Tab 4: Replicated Logs at every MSS
Tab 5: Live Queue States after service
"""

import streamlit as st
import pandas as pd
import numpy as np
import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import plotly.graph_objects as go

# ═══════════════════════════════════════════════════════════════
#                        MODEL CLASSES
# ═══════════════════════════════════════════════════════════════

class LamportClock:
    def __init__(self): self.time = 0
    def tick(self): self.time += 1; return self.time
    def merge(self, t): self.time = max(self.time, t) + 1; return self.time


@dataclass
class Request:
    mh_id: str; source_mss: int; priority: int; timestamp: int
    status: str = "PENDING"
    rid: str = field(default="", init=False)
    def __post_init__(self): self.rid = f"REQ_{self.mh_id}_T{self.timestamp}"
    def row(self):
        return {"ID": self.rid, "MH": self.mh_id, "MSS": f"MSS_{self.source_mss}",
                "Priority": self.priority, "Lamport T": self.timestamp, "Status": self.status}


class MobileHost:
    def __init__(self, name, mss, pri=5):
        self.id = name; self.mss = mss; self.pri = pri
        self.req: Optional[Request] = None
    def request_cs(self):
        if self.req and self.req.status == "PENDING": return None
        r = self.mss.new_request(self, self.pri); self.req = r; return r


class MSS:
    def __init__(self, i):
        self.id = i; self.nxt: Optional["MSS"] = None
        self.has_token = False; self.clock = LamportClock()
        self.rep_log: List[Request] = []; self.local_q: List[Request] = []
        self.global_q: List[Request] = []; self.hosts: List[MobileHost] = []
        self.tx = 0; self.rx = 0; self.grants = 0

    def add(self, mh): self.hosts.append(mh); mh.mss = self

    def new_request(self, mh, pri):
        t = self.clock.tick()
        r = Request(mh.id, self.id, pri, t)
        self.rep_log.append(r); self.local_q.append(r)
        self.global_q.append(r); self._sort()
        self._broadcast(r); return r

    def _broadcast(self, r):
        c = self.nxt
        while c and c.id != self.id:
            c.recv(r); self.tx += 1; c = c.nxt

    def recv(self, r):
        self.clock.merge(r.timestamp)
        self.rep_log.append(r); self.global_q.append(r)
        self._sort(); self.rx += 1

    def _sort(self):
        self.global_q.sort(key=lambda r: (-r.priority, r.timestamp))

    def try_grant(self):
        if not self.has_token: return None
        lp = [r for r in self.global_q if r.source_mss == self.id and r.status == "PENDING"]
        if not lp: return None
        g = lp[0]; g.status = "GRANTED"
        if g in self.local_q: self.local_q.remove(g)
        self.grants += 1; return g

    def pass_token(self):
        if not self.has_token: return
        self.has_token = False; self.nxt.has_token = True
        self.nxt.clock.tick(); self.tx += 1

    def stats(self):
        return {"MSS": f"MSS_{self.id}", "Sent": self.tx, "Recv": self.rx,
                "Grants": self.grants, "MHs": len(self.hosts),
                "Pending": sum(1 for r in self.local_q if r.status == "PENDING")}


class Ring:
    def __init__(self, n):
        self.n = n; self.nodes = [MSS(i) for i in range(n)]
        for i in range(n):
            self.nodes[i].nxt = self.nodes[(i + 1) % n]
        self.nodes[0].has_token = True
    def holder(self):
        for m in self.nodes:
            if m.has_token: return m
        return None


class Engine:
    def __init__(self, ring):
        self.ring = ring; self.log: List[str] = []
    def step(self):
        h = self.ring.holder()
        if not h: return None, "⚠️ no holder"
        g = h.try_grant()
        if g:
            msg = f"✅ MSS_{h.id} GRANTED → {g.mh_id} (P={g.priority})"
            self.log.append(msg); return g, msg
        nid = h.nxt.id; h.pass_token()
        msg = f"➡️ Token: MSS_{h.id} → MSS_{nid}"; self.log.append(msg)
        return None, msg


def build_world(n=4):
    ring = Ring(n); mhs = []
    for i in range(n):
        for j in range(3):
            m = MobileHost(f"MH_{i}_{chr(65+j)}", ring.nodes[i], 5+i+j)
            ring.nodes[i].add(m); mhs.append(m)
    return ring, mhs


# ═══════════════════════════════════════════════════════════════
#             GEOMETRY HELPER  (shared by all animators)
# ═══════════════════════════════════════════════════════════════

class Geom:
    def __init__(self, n, r=2.2, mr=0.55):
        self.n, self.R, self.mr = n, r, mr
        self.ang = [2*math.pi*i/n - math.pi/2 for i in range(n)]
        self.sx = [r*math.cos(a) for a in self.ang]
        self.sy = [r*math.sin(a) for a in self.ang]
        self.hp, self.hn = {}, {}
        for s in range(n):
            self.hp[s], self.hn[s] = [], []
            for j in range(3):
                a2 = self.ang[s]+math.pi+(j-1)*0.4
                self.hp[s].append((self.sx[s]+mr*math.cos(a2), self.sy[s]+mr*math.sin(a2)))
                self.hn[s].append(f"MH_{s}_{chr(65+j)}")


def lerp(a, b, t): return a+(b-a)*max(0., min(1., t))


# ═══════════════════════════════════════════════════════════════
#               MAIN TOKEN-RING ANIMATION BUILDER
# ═══════════════════════════════════════════════════════════════

class MainAnim:
    """Native-JS Plotly animation; constant trace count → zero flicker."""
    C = dict(tf='#FFF', th='#0F0', mn='#00D4FF', mp='#FFD700', mh='#0F0',
             mg='#CC00FF', mr='#FF5722', mc='#9C27B0', ck='#87CEEB',
             ring='#444', conn='#666')

    def __init__(self, g: Geom, req_mh, ho_mh):
        self.g = g; self.req_mh = req_mh; self.ho_mh = ho_mh
        self.rms = int(req_mh.split('_')[1]); self.rmi = ord(req_mh.split('_')[2])-65
        self.hms = int(ho_mh.split('_')[1]); self.hmi = ord(ho_mh.split('_')[2])-65
        self.htgt = (self.hms+1) % g.n
        self.frames = []; self.logs = []
        self.tp = 0.0; self.MS = 10; self.ST = 8; self.HL = 15; self.MG = 8

    # ── frame builder (always 11 + n traces) ──────────────────
    def _frame(self, tc, sc, hc, rm=None, pm=None, rl=None, hl=None, txt="", ovr=None):
        C, g = self.C, self.g; d = []
        # 0  ring
        ra = np.linspace(0, 2*np.pi, 100)
        d.append(go.Scatter(x=g.R*1.1*np.cos(ra), y=g.R*1.1*np.sin(ra),
                            mode='lines', line=dict(color=C['ring'], width=3), hoverinfo='none'))
        # 1..n  arrows
        for i in range(g.n):
            m = (g.ang[i]+g.ang[(i+1)%g.n])/2
            if i == g.n-1: m = (g.ang[i]+g.ang[0]+2*math.pi)/2
            ax, ay = g.R*1.1*math.cos(m), g.R*1.1*math.sin(m)
            d.append(go.Scatter(x=[ax, ax+.12*math.cos(m+math.pi/2)],
                                y=[ay, ay+.12*math.sin(m+math.pi/2)],
                                mode='lines', line=dict(color=C['ring'], width=2), hoverinfo='none'))
        # n+1  connections
        lx, ly = [], []
        for s in range(g.n):
            for j in range(3):
                nm = g.hn[s][j]
                if ovr and nm in ovr: continue
                lx += [g.sx[s], g.hp[s][j][0], None]; ly += [g.sy[s], g.hp[s][j][1], None]
        d.append(go.Scatter(x=lx, y=ly, mode='lines',
                            line=dict(color=C['conn'], width=1, dash='dot'), hoverinfo='none'))
        # n+2  MSS
        mc = [sc.get(i, C['mn']) for i in range(g.n)]
        ms = [55 if i in sc else 48 for i in range(g.n)]
        d.append(go.Scatter(x=g.sx, y=g.sy, mode='markers+text',
                            marker=dict(size=ms, color=mc, line=dict(width=3, color='white'), symbol='square'),
                            text=[f'MSS_{i}' for i in range(g.n)], textposition='top center',
                            textfont=dict(color='white', size=11), hoverinfo='none'))
        # n+3  MHs
        hx, hy, hcc, hs, ht = [], [], [], [], []
        for s in range(g.n):
            for j in range(3):
                nm = g.hn[s][j]
                mx, my = ovr[nm] if (ovr and nm in ovr) else g.hp[s][j]
                hx.append(mx); hy.append(my); ht.append(nm.split('_')[-1])
                hcc.append(hc.get(nm, '#4CAF50')); hs.append(28 if nm in hc else 22)
        d.append(go.Scatter(x=hx, y=hy, mode='markers+text',
                            marker=dict(size=hs, color=hcc, line=dict(width=2, color='white'), symbol='circle'),
                            text=ht, textposition='bottom center', textfont=dict(color='white', size=10), hoverinfo='none'))
        # n+4,5,6  messages
        def _msg(obj, lbl, col):
            if obj: d.append(go.Scatter(x=[obj['x']], y=[obj['y']], opacity=1, mode='markers+text',
                                        text=[lbl], textposition='top center', textfont=dict(color='white', size=8),
                                        marker=dict(size=15, color=col, symbol='square',
                                                    line=dict(width=2, color='white')), hoverinfo='none'))
            else:    d.append(go.Scatter(x=[0], y=[0], opacity=0, hoverinfo='none'))
        _msg(rm, 'REQ', '#FF5722'); _msg(pm, 'PERM', '#9C27B0'); _msg(rl, 'REL', '#2196F3')
        # n+7  handoff
        if hl: d.append(go.Scatter(x=[hl['x1'], hl['x2']], y=[hl['y1'], hl['y2']],
                                   mode='lines', line=dict(color='#FF9800', width=3, dash='dash'), hoverinfo='none'))
        else:  d.append(go.Scatter(x=[0], y=[0], opacity=0, hoverinfo='none'))
        # n+8  token
        ta = ((self.tp % g.n)/g.n)*2*math.pi - math.pi/2
        d.append(go.Scatter(x=[g.R*math.cos(ta)], y=[g.R*math.sin(ta)], mode='markers+text',
                            marker=dict(size=35, color=tc, symbol='circle', line=dict(width=4, color='#333')),
                            text=['🔑'], textfont=dict(size=14), hoverinfo='none'))
        # n+9  log text
        d.append(go.Scatter(x=[0], y=[-3.3], mode='text',
                            text=[f'<b>{txt}</b>'], textfont=dict(size=13, color='white'), hoverinfo='none'))
        return d

    # ── movement helpers ──────────────────────────────────────
    def _to(self, tgt, **kw):
        cur = self.tp % self.g.n; diff = (tgt - cur) % self.g.n
        if diff < 0.01: return
        s = self.tp; e = self.tp + diff
        for i in range(self.MS):
            self.tp = lerp(s, e, (i+1)/self.MS); self.frames.append(self._frame(**kw))

    def _hold(self, n, **kw):
        for _ in range(n): self.frames.append(self._frame(**kw))

    def _msg(self, fp, tp, key, **kw):
        for i in range(self.MG):
            t = (i+1)/self.MG
            kw[key] = {'x': lerp(fp[0], tp[0], t), 'y': lerp(fp[1], tp[1], t)}
            self.frames.append(self._frame(**kw))

    # ── full scenario ─────────────────────────────────────────
    def generate(self):
        C, g = self.C, self.g
        rx, ry = g.hp[self.rms][self.rmi]; smx, smy = g.sx[self.rms], g.sy[self.rms]
        hx, hy = g.hp[self.hms][self.hmi]; hmx, hmy = g.sx[self.hms], g.sy[self.hms]
        nx, ny = g.hp[self.htgt][0]; nmx, nmy = g.sx[self.htgt], g.sy[self.htgt]

        def L(s): self.logs.append(s)

        # ── Phase 1: request ──
        L(f"Phase 1 · {self.req_mh} sends REQUEST to MSS_{self.rms}")
        self._msg((rx,ry),(smx,smy),'rm', tc=C['tf'], sc={}, hc={self.req_mh:C['mr']},
                  txt=f"📤 {self.req_mh} → REQUEST → MSS_{self.rms}")

        # ── Phase 2: queue ──
        L(f"Phase 2 · MSS_{self.rms} queues request, BROADCASTS to all other MSSs")
        self._hold(self.ST, tc=C['tf'], sc={self.rms:C['mp']}, hc={self.req_mh:C['mr']},
                   txt=f"📋 MSS_{self.rms} queued request · Broadcasting to {g.n-1} MSSs")

        # ── Phase 3: token circulation ──
        L(f"Phase 3 · Token circulates from MSS_0, briefly checking each MSS")
        for node in range(1, self.rms+1):
            L(f"  └─ Token visits MSS_{node}" + (" ← has pending request!" if node==self.rms else ""))
            self._to(node, tc=C['tf'], sc={self.rms:C['mp']}, hc={self.req_mh:C['mr']},
                     txt=f"⚪ Token → MSS_{node}")
            if node != self.rms:
                self._hold(self.ST, tc=C['tf'],
                           sc={node:C['ck'], self.rms:C['mp']},
                           hc={self.req_mh:C['mr']},
                           txt=f"🔍 MSS_{node}: queue empty → pass token")

        # ── Phase 4: hold ──
        L(f"Phase 4 · MSS_{self.rms} HOLDS token — pending request from {self.req_mh}")
        self._hold(self.HL, tc=C['th'], sc={self.rms:C['mh']}, hc={self.req_mh:C['mr']},
                   txt=f"🟢 MSS_{self.rms} HOLDS token!")

        # ── Phase 5: permission ──
        L(f"Phase 5 · MSS_{self.rms} grants PERMISSION to {self.req_mh} (highest local priority)")
        self._msg((smx,smy),(rx,ry),'pm', tc=C['th'], sc={self.rms:C['mg']}, hc={self.req_mh:C['mr']},
                  txt=f"📨 PERMISSION → {self.req_mh}")

        # ── Phase 6: CS ──
        L(f"Phase 6 · {self.req_mh} enters CRITICAL SECTION")
        self._hold(self.HL, tc=C['th'], sc={self.rms:C['mh']}, hc={self.req_mh:C['mc']},
                   txt=f"🟣 {self.req_mh} in CS")

        # ── Phase 7: release ──
        L(f"Phase 7 · {self.req_mh} finishes CS, sends RELEASE")
        self._msg((rx,ry),(smx,smy),'rl', tc=C['th'], sc={self.rms:C['mh']}, hc={self.req_mh:C['mc']},
                  txt=f"📤 RELEASE → MSS_{self.rms}")

        # ── Phase 8: second request ──
        L(f"Phase 8 · {self.ho_mh} sends REQUEST to MSS_{self.hms}")
        self._msg((hx,hy),(hmx,hmy),'rm', tc=C['tf'], sc={}, hc={self.ho_mh:C['mr']},
                  txt=f"📤 {self.ho_mh} → REQUEST → MSS_{self.hms}")
        self._hold(self.ST, tc=C['tf'], sc={self.hms:C['mp']}, hc={self.ho_mh:C['mr']},
                   txt=f"📋 MSS_{self.hms} queued · Broadcasting")

        # ── Phase 9: handoff movement ──
        L(f"Phase 9 · HANDOFF: {self.ho_mh} moves from MSS_{self.hms} → MSS_{self.htgt}")
        for i in range(self.MG*2):
            t = (i+1)/(self.MG*2)
            cx, cy = lerp(hx, nx, t), lerp(hy, ny, t)
            self.tp += 0.5/(self.MG*2)
            self.frames.append(self._frame(
                tc=C['tf'], sc={self.hms:C['mp']}, hc={self.ho_mh:'#FF9800'},
                hl={'x1':hx,'y1':hy,'x2':cx,'y2':cy},
                txt=f"📱 HANDOFF → MSS_{self.htgt}",
                ovr={self.ho_mh:(cx,cy)}))

        # ── Phase 10: kill ──
        L(f"Phase 10 · Request KILLED at MSS_{self.hms} — MH departed")
        for i in range(self.HL):
            self.frames.append(self._frame(
                tc=C['tf'], sc={self.hms:'#F00' if i%6<3 else C['mn']}, hc={},
                txt=f"❌ Request KILLED at MSS_{self.hms}",
                ovr={self.ho_mh:(nx,ny)}))

        # ── Phase 11: re-register ──
        L(f"Phase 11 · {self.ho_mh} re-registers at MSS_{self.htgt}")
        self._msg((nx,ny),(nmx,nmy),'rm', tc=C['tf'], sc={}, hc={self.ho_mh:C['mr']},
                  txt=f"📤 RE-REGISTER at MSS_{self.htgt}", ovr={self.ho_mh:(nx,ny)})
        self._hold(self.ST, tc=C['tf'], sc={self.htgt:C['mp']}, hc={self.ho_mh:C['mr']},
                   txt=f"📋 MSS_{self.htgt} queued new request", ovr={self.ho_mh:(nx,ny)})

        # ── Phase 12: token continues ──
        L(f"Phase 12 · Token continues checking intermediate MSSs")
        cid = int(self.tp)+1; safety = 0
        while cid % g.n != self.htgt and safety < g.n+2:
            tgt = cid % g.n
            L(f"  └─ MSS_{tgt}: queue empty → pass")
            self._to(tgt, tc=C['tf'], sc={self.htgt:C['mp']}, hc={self.ho_mh:C['mr']},
                     txt=f"⚪ Token → MSS_{tgt}", ovr={self.ho_mh:(nx,ny)})
            self._hold(self.ST, tc=C['tf'],
                       sc={tgt:C['ck'], self.htgt:C['mp']},
                       hc={self.ho_mh:C['mr']},
                       txt=f"🔍 MSS_{tgt}: empty → pass",
                       ovr={self.ho_mh:(nx,ny)})
            cid += 1; safety += 1

        # ── Phase 13: arrive & grant ──
        L(f"Phase 13 · Token arrives at MSS_{self.htgt}, grants to {self.ho_mh}")
        self._to(self.htgt, tc=C['tf'], sc={self.htgt:C['mp']}, hc={self.ho_mh:C['mr']},
                 txt=f"⚪ Token → MSS_{self.htgt}", ovr={self.ho_mh:(nx,ny)})
        self._hold(self.HL, tc=C['th'], sc={self.htgt:C['mh']}, hc={self.ho_mh:C['mr']},
                   txt=f"🟢 MSS_{self.htgt} HOLDS token!", ovr={self.ho_mh:(nx,ny)})
        self._msg((nmx,nmy),(nx,ny),'pm', tc=C['th'], sc={self.htgt:C['mg']}, hc={self.ho_mh:C['mr']},
                  txt=f"📨 PERMISSION → {self.ho_mh}", ovr={self.ho_mh:(nx,ny)})

        # ── Phase 14: CS at new MSS ──
        L(f"Phase 14 · {self.ho_mh} enters CS at new MSS_{self.htgt}")
        self._hold(self.HL, tc=C['th'], sc={self.htgt:C['mh']}, hc={self.ho_mh:C['mc']},
                   txt=f"🟣 {self.ho_mh} in CS (new MSS)", ovr={self.ho_mh:(nx,ny)})

        # ── Phase 15: resume ──
        L("Phase 15 · Normal operation resumes")
        for i in range(self.MS*2):
            self.tp += 1.0/(self.MS)
            self.frames.append(self._frame(tc=C['tf'], sc={}, hc={},
                               txt="⚪ Normal circulation…"))

        # ── package ──
        pf = [go.Frame(data=f, name=str(i)) for i, f in enumerate(self.frames)]
        lo = go.Layout(
            title=dict(text='<b>Token Ring Animation (Native JS — Zero Flicker)</b>',
                       font=dict(color='white', size=16), x=0.5),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-3.5,3.5]),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False,
                       range=[-3.8,3.5], scaleanchor='x'),
            plot_bgcolor='#1a1a2e', paper_bgcolor='#1a1a2e', height=700,
            margin=dict(l=20,r=20,t=60,b=20), showlegend=False,
            updatemenus=[dict(type='buttons', showactive=False, x=.05, y=-.05,
                xanchor='left', yanchor='top', direction='right',
                buttons=[
                    dict(label='▶ Play', method='animate',
                         args=[None, dict(frame=dict(duration=50,redraw=True),
                                         fromcurrent=True, transition=dict(duration=0))]),
                    dict(label='⏸ Pause', method='animate',
                         args=[[None], dict(frame=dict(duration=0,redraw=False),
                                           mode='immediate')])])],
            sliders=[dict(active=0, yanchor='top', xanchor='left',
                currentvalue=dict(font=dict(size=12, color='white'),
                                  prefix='Frame: ', visible=True),
                transition=dict(duration=0), pad=dict(b=10,t=30),
                len=.8, x=.2, y=-.05,
                steps=[dict(args=[[str(i)], dict(frame=dict(duration=0,redraw=True),
                            mode='immediate')], label='', method='animate')
                       for i in range(len(pf))])])
        return self.frames[0], pf, lo, self.logs


# ═══════════════════════════════════════════════════════════════
#             BROADCAST ANIMATION BUILDER
# ═══════════════════════════════════════════════════════════════

class BcastAnim:
    """Shows request broadcast: MH→MSS→ALL other MSSs, native JS."""
    def __init__(self, g: Geom, src_mss, src_mh_idx, priority):
        self.g = g; self.src = src_mss; self.mi = src_mh_idx
        self.pri = priority; self.frames = []; self.MG = 8; self.ST = 6

    def _f(self, mc, mh_hl=None, msg=None, ax=None, ay=None, qc=None, txt=""):
        g = self.g; d = []
        # ring
        ra = np.linspace(0,2*np.pi,100)
        d.append(go.Scatter(x=g.R*1.05*np.cos(ra), y=g.R*1.05*np.sin(ra),
                            mode='lines', line=dict(color='#555',width=2), hoverinfo='none'))
        # MSS
        cols = [mc.get(i,'#00D4FF') for i in range(g.n)]
        sz = [55 if i in mc else 45 for i in range(g.n)]
        d.append(go.Scatter(x=g.sx, y=g.sy, mode='markers+text',
                            marker=dict(size=sz,color=cols,line=dict(width=3,color='white'),symbol='square'),
                            text=[f'MSS_{i}' for i in range(g.n)], textposition='top center',
                            textfont=dict(size=11,color='black'), hoverinfo='none'))
        # MH
        hx,hy,hcc,hs = [],[],[],[]
        for s in range(g.n):
            for j in range(3):
                hx.append(g.hp[s][j][0]); hy.append(g.hp[s][j][1])
                nm = g.hn[s][j]
                if mh_hl and nm == mh_hl: hcc.append('#FF5722'); hs.append(28)
                else: hcc.append('#4CAF50'); hs.append(20)
        d.append(go.Scatter(x=hx,y=hy,mode='markers',
                            marker=dict(size=hs,color=hcc,line=dict(width=1,color='white')),hoverinfo='none'))
        # arrows
        d.append(go.Scatter(x=ax or [],y=ay or [],mode='lines',
                            line=dict(color='#FF5722',width=2.5),hoverinfo='none'))
        # msg
        if msg: d.append(go.Scatter(x=[msg[0]],y=[msg[1]],mode='markers+text',
                    marker=dict(size=18,color='#FF5722',symbol='diamond',line=dict(width=2,color='white')),
                    text=['📧'],textfont=dict(size=12),hoverinfo='none',opacity=1))
        else:   d.append(go.Scatter(x=[0],y=[0],opacity=0,hoverinfo='none'))
        # queue badges
        bx,by,bt = [],[],[]
        for i in range(g.n):
            bx.append(g.sx[i]); by.append(g.sy[i]-.35)
            bt.append(f"Q:{qc.get(i,0)}" if qc and qc.get(i,0)>0 else "")
        d.append(go.Scatter(x=bx,y=by,mode='text',text=bt,
                            textfont=dict(size=11,color='#FFD700',family='Arial Black'),hoverinfo='none'))
        # log
        d.append(go.Scatter(x=[0],y=[-2.9],mode='text',text=[f'<b>{txt}</b>'],
                            textfont=dict(size=13,color='#333'),hoverinfo='none'))
        return d

    def build(self):
        g = self.g; mh = g.hn[self.src][self.mi]
        mx,my = g.hp[self.src][self.mi]; sx,sy = g.sx[self.src],g.sy[self.src]
        ax,ay,qc = [],[],{}

        # A  request
        for i in range(self.MG):
            t=(i+1)/self.MG
            self.frames.append(self._f(mc={},mh_hl=mh,
                msg=(lerp(mx,sx,t),lerp(my,sy,t)),
                txt=f"📤 {mh} (P={self.pri}) → MSS_{self.src}"))
        # B  queue
        qc[self.src]=1
        for _ in range(self.ST):
            self.frames.append(self._f(mc={self.src:'#FFD700'},mh_hl=mh,qc=dict(qc),
                txt=f"📋 MSS_{self.src} queued request"))
        # C  broadcast
        recv = set()
        for tgt in range(g.n):
            if tgt==self.src: continue
            tx,ty = g.sx[tgt],g.sy[tgt]
            for i in range(self.MG):
                t=(i+1)/self.MG
                self.frames.append(self._f(
                    mc={self.src:'#FFD700', **{m:'#87CEEB' for m in recv}},
                    mh_hl=mh, msg=(lerp(sx,tx,t),lerp(sy,ty,t)),
                    ax=list(ax),ay=list(ay),qc=dict(qc),
                    txt=f"📡 Broadcasting → MSS_{tgt}…"))
            ax += [sx,tx,None]; ay += [sy,ty,None]
            recv.add(tgt); qc[tgt]=1
            for _ in range(self.ST):
                self.frames.append(self._f(
                    mc={self.src:'#FFD700',tgt:'#00FF00',**{m:'#87CEEB' for m in recv if m!=tgt}},
                    mh_hl=mh,ax=list(ax),ay=list(ay),qc=dict(qc),
                    txt=f"✅ MSS_{tgt} received & replicated!"))
        # D  done
        for _ in range(self.ST*2):
            self.frames.append(self._f(mc={i:'#00FF00' for i in range(g.n)},
                mh_hl=mh,ax=list(ax),ay=list(ay),qc=dict(qc),
                txt=f"✅ ALL {g.n} MSSs now hold {mh}'s request!"))

        pf = [go.Frame(data=f,name=str(i)) for i,f in enumerate(self.frames)]
        lo = go.Layout(
            title=dict(text='<b>Request Broadcasting: MH → MSS → ALL MSSs</b>',
                       font=dict(size=15),x=.5),
            xaxis=dict(showgrid=False,zeroline=False,showticklabels=False,range=[-3.2,3.2]),
            yaxis=dict(showgrid=False,zeroline=False,showticklabels=False,range=[-3.3,3.2],scaleanchor='x'),
            height=620, plot_bgcolor='#f5f5f5', paper_bgcolor='#f5f5f5',
            margin=dict(l=20,r=20,t=60,b=20), showlegend=False,
            updatemenus=[dict(type='buttons',showactive=False,x=.05,y=-.05,
                xanchor='left',yanchor='top',direction='right',
                buttons=[
                    dict(label='▶ Play',method='animate',
                         args=[None,dict(frame=dict(duration=60,redraw=True),fromcurrent=True,
                                        transition=dict(duration=0))]),
                    dict(label='⏸ Pause',method='animate',
                         args=[[None],dict(frame=dict(duration=0,redraw=False),mode='immediate')])])],
            sliders=[dict(active=0,yanchor='top',xanchor='left',
                currentvalue=dict(font=dict(size=11),prefix='Step: ',visible=True),
                transition=dict(duration=0),pad=dict(b=10,t=30),len=.8,x=.2,y=-.05,
                steps=[dict(args=[[str(i)],dict(frame=dict(duration=0,redraw=True),mode='immediate')],
                       label='',method='animate') for i in range(len(pf))])])
        return self.frames[0], pf, lo


# ═══════════════════════════════════════════════════════════════
#                 PRIORITY CHART HELPER
# ═══════════════════════════════════════════════════════════════

def priority_bar(reqs: List[Request], holder_id: int):
    """Horizontal bar chart of queue sorted by priority; grantee in green."""
    if not reqs: return None
    s = sorted(reqs, key=lambda r: (-r.priority, r.timestamp))
    grantee = None
    for r in s:
        if r.source_mss == holder_id and r.status == "PENDING":
            grantee = r; break
    cols = []
    for r in s:
        if r is grantee:        cols.append('#00CC00')
        elif r.status=="GRANTED":  cols.append('#9C27B0')
        elif r.status=="COMPLETED":cols.append('#888')
        else:                      cols.append('#FF5722')
    fig = go.Figure(go.Bar(
        y=[f"{r.mh_id} @ MSS_{r.source_mss}" for r in s],
        x=[r.priority for r in s], orientation='h',
        marker=dict(color=cols, line=dict(width=1,color='#333')),
        text=[f"P={r.priority} T={r.timestamp} [{r.status}]" for r in s],
        textposition='inside', textfont=dict(color='white',size=11)))
    fig.update_layout(
        title=f"<b>Global Queue at Token Holder MSS_{holder_id}</b><br>"
              f"<sub>🟢 = Will be granted next  |  🟠 = Pending  |  🟣 = Granted  |  ⚫ = Done</sub>",
        xaxis_title="Priority", yaxis=dict(autorange='reversed'),
        height=max(280, len(s)*48+120), margin=dict(l=150,r=30,t=80,b=40),
        plot_bgcolor='#fafafa')
    return fig


def static_ring(ring):
    """Static ring diagram for context."""
    g = Geom(ring.n, r=2.0, mr=0.5)
    fig = go.Figure()
    ra = np.linspace(0,2*np.pi,100)
    fig.add_trace(go.Scatter(x=g.R*1.05*np.cos(ra),y=g.R*1.05*np.sin(ra),
                             mode='lines',line=dict(color='#bbb',width=2),hoverinfo='none',showlegend=False))
    cols = ['gold' if m.has_token else '#64b5f6' for m in ring.nodes]
    fig.add_trace(go.Scatter(x=g.sx,y=g.sy,mode='markers+text',
        marker=dict(size=[55 if m.has_token else 45 for m in ring.nodes],
                    color=cols,line=dict(width=2,color='#333'),symbol='square'),
        text=[f'MSS_{m.id}' for m in ring.nodes],textposition='top center',
        textfont=dict(size=12),hoverinfo='none',showlegend=False))
    for s in range(ring.n):
        for j,mh in enumerate(ring.nodes[s].hosts):
            mx,my = g.hp[s][j]
            fig.add_trace(go.Scatter(x=[g.sx[s],mx],y=[g.sy[s],my],mode='lines',
                line=dict(width=1,color='#ccc',dash='dot'),hoverinfo='none',showlegend=False))
            fig.add_trace(go.Scatter(x=[mx],y=[my],mode='markers+text',
                text=[mh.id.split('_')[-1]],textposition='bottom center',textfont=dict(size=9),
                marker=dict(size=18,color='#4CAF50',line=dict(width=1,color='white')),
                hoverinfo='text',hovertext=f'{mh.id} P={mh.pri}',showlegend=False))
    fig.update_layout(xaxis=dict(showgrid=False,zeroline=False,showticklabels=False,range=[-3.2,3.2]),
                      yaxis=dict(showgrid=False,zeroline=False,showticklabels=False,
                                 range=[-3.2,3.2],scaleanchor='x'),
                      height=450,margin=dict(l=10,r=10,t=30,b=10),
                      plot_bgcolor='white',paper_bgcolor='white')
    return fig


# ═══════════════════════════════════════════════════════════════
#                       STREAMLIT APP
# ═══════════════════════════════════════════════════════════════

st.set_page_config(page_title='Token-Ring ME', page_icon='🔐', layout='wide')
st.markdown("""<style>.block-container{padding-top:.8rem}
.hdr{font-size:1.8rem;font-weight:700;color:#0d47a1;text-align:center;padding:.7rem;
background:linear-gradient(90deg,#e3f2fd,#bbdefb);border-radius:10px;margin-bottom:1rem}
</style><div class="hdr">🔐 Token-Ring Mutual Exclusion — Replication Scheme</div>""",
            unsafe_allow_html=True)

if 'ring' not in st.session_state:
    r, m = build_world(4)
    st.session_state.update(ring=r, mhs=m, eng=Engine(r), step=0, reqs=[])

ring: Ring = st.session_state.ring
mhs  = st.session_state.mhs
eng: Engine = st.session_state.eng

# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.header('⚙️ Controls')
    if st.button('🔄 Reset', use_container_width=True):
        for k in list(st.session_state.keys()): del st.session_state[k]
        st.rerun()
    st.markdown('---')
    st.subheader('📤 Send Request')
    by_mss = {}
    for m in mhs: by_mss.setdefault(m.mss.id,[]).append(m)
    sel_mss = st.selectbox('MSS', range(ring.n), format_func=lambda i: f'MSS_{i}')
    at = by_mss.get(sel_mss,[])
    if at:
        si = st.selectbox('MH',range(len(at)),format_func=lambda i: f'{at[i].id} (P={at[i].pri})')
        if st.button('📤 Send', use_container_width=True):
            r = at[si].request_cs()
            if r: st.session_state.reqs.append(r); st.success(f'{at[si].id} requested')
    st.markdown('---')
    st.subheader('🔄 Token')
    c1,c2 = st.columns(2)
    with c1:
        if st.button('▶ Step', use_container_width=True): eng.step(); st.session_state.step += 1
    with c2:
        if st.button('⏩ ×5', use_container_width=True):
            for _ in range(5): eng.step(); st.session_state.step += 1
    st.markdown('---')
    h = ring.holder()
    st.metric('Steps', st.session_state.step)
    st.metric('Token', f'MSS_{h.id}' if h else '—')


# ── Tabs ──────────────────────────────────────────────────────
t1,t2,t3,t4,t5 = st.tabs(['🎬 Animation','📡 Broadcasting','🏆 Priority','📋 Logs','📊 Queues'])

# ═════════════════ TAB 1: ANIMATION ═══════════════════════════
with t1:
    st.markdown('### Full Token Ring Animation')
    st.info('Uses **native Plotly JS** — no Streamlit re-rendering, no flickering. '
            'Token position is tracked with unwrapped math — no jumping.')
    c1,c2,c3 = st.columns(3)
    nm = c1.selectbox('MSSs',[4,5,6],index=2,key='an')
    opts = [f"MH_{i}_{chr(65+j)}" for i in range(nm) for j in range(3)]
    rq = c2.selectbox('Requesting MH',opts,index=min(6,len(opts)-1),key='rq')
    ho = c3.selectbox('Handoff MH',[o for o in opts if o!=rq],index=min(4,len(opts)-3),key='ho')

    if st.button('🎬 Generate',type='primary',use_container_width=True):
        with st.spinner('Building…'):
            g = Geom(nm); a = MainAnim(g, rq, ho); d0,pf,lo,lg = a.generate()
            st.session_state.anim = (d0,pf,lo,lg)

    if 'anim' in st.session_state:
        d0,pf,lo,lg = st.session_state.anim
        st.plotly_chart(go.Figure(data=d0,frames=pf,layout=lo), use_container_width=True)
        st.markdown('#### 📖 Phase Log')
        for i,e in enumerate(lg,1):
            if e.startswith("Phase"):
                st.markdown(f"**{i}. {e}**")
            else:
                st.caption(f"  {e}")

# ═════════════════ TAB 2: BROADCASTING ════════════════════════
with t2:
    st.markdown('### 📡 Request Broadcasting to ALL MSSs')
    st.info('When an MH sends a request, its local MSS **replicates** the request '
            'to **every other MSS** in the ring. This ensures all MSSs have a '
            'consistent view of pending requests.')

    bc1,bc2,bc3 = st.columns(3)
    bnm = bc1.selectbox('Ring size',[4,5,6],index=1,key='bn')
    bopts = [f"MH_{i}_{chr(65+j)}" for i in range(bnm) for j in range(3)]
    bmh = bc2.selectbox('Source MH',bopts,index=3,key='bmh')
    bpri = bc3.slider('Priority',1,10,7,key='bpri')

    bs = int(bmh.split('_')[1]); bi = ord(bmh.split('_')[2])-65

    if st.button('📡 Generate Broadcast Animation',type='primary',use_container_width=True):
        with st.spinner('Building broadcast frames…'):
            bg = Geom(bnm); ba = BcastAnim(bg, bs, bi, bpri)
            d0,pf,lo = ba.build()
            st.session_state.bcast = (d0,pf,lo)

    if 'bcast' in st.session_state:
        d0,pf,lo = st.session_state.bcast
        st.plotly_chart(go.Figure(data=d0,frames=pf,layout=lo),use_container_width=True)

    st.markdown('---')
    st.markdown('#### How Broadcasting Works')
    st.markdown(f'''
    1. **{bmh}** sends a `REQUEST(priority={bpri})` to its local **MSS_{bs}**
    2. **MSS_{bs}** adds the request to its **local queue** and **replicated log**
    3. **MSS_{bs}** sends the request to **every other MSS** in the ring:
    ''')
    for i in range(bnm):
        if i != bs:
            st.markdown(f'   - MSS\_{bs} → **MSS\_{i}** ✉️')
    st.markdown(f'''
    4. Each receiving MSS adds the request to its own **replicated log** and **global queue**
    5. Result: **All {bnm} MSSs** now know about {bmh}'s request

    > This replication ensures that whichever MSS holds the token can make an
    > informed priority-based decision about granting access.
    ''')

# ═════════════════ TAB 3: PRIORITY & GRANTING ═════════════════
with t3:
    st.markdown('### 🏆 Priority-Based Granting')
    st.info('When the token arrives at an MSS, it checks the **global queue** for '
            'local pending requests and grants to the one with **highest priority**. '
            'Use the sidebar to send requests from different MHs, then step the token.')

    holder = ring.holder()
    if holder:
        st.markdown(f'#### Token is at **MSS_{holder.id}** 🔑')

        # Collect all pending requests across all MSSs
        all_pending = []
        seen = set()
        for mss in ring.nodes:
            for r in mss.global_q:
                if r.rid not in seen:
                    all_pending.append(r); seen.add(r.rid)

        if all_pending:
            fig = priority_bar(all_pending, holder.id)
            if fig:
                st.plotly_chart(fig, use_container_width=True)

            st.markdown('---')
            st.markdown('#### Granting Logic')
            local_pending = [r for r in all_pending
                             if r.source_mss == holder.id and r.status == "PENDING"]
            if local_pending:
                best = max(local_pending, key=lambda r: (r.priority, -r.timestamp))
                st.success(f'**Next grant →** {best.mh_id} (Priority **{best.priority}**, '
                           f'Timestamp {best.timestamp})')

                st.markdown(f'''
                **Why {best.mh_id}?**
                - Token is at MSS\_{holder.id}
                - {len(local_pending)} local pending request(s) found
                - Sorted by **(priority DESC, timestamp ASC)**
                - {best.mh_id} has the **highest priority** among local requests
                ''')
            else:
                st.warning(f'No local pending requests at MSS_{holder.id} — '
                           f'token will **pass** to MSS_{holder.nxt.id}')
                remote = [r for r in all_pending if r.status == "PENDING"]
                if remote:
                    st.caption(f'{len(remote)} pending request(s) at other MSSs — '
                               f'token must circulate to reach them.')

            st.markdown('---')
            st.markdown('#### All Pending Requests')
            df = pd.DataFrame([r.row() for r in all_pending])
            st.dataframe(df, use_container_width=True, hide_index=True)
        else:
            st.success('No pending requests in the system.')

        st.markdown('---')
        st.markdown('#### Quick Actions')
        qc1,qc2 = st.columns(2)
        with qc1:
            if st.button('▶ Step Token (1×)', key='pstep', use_container_width=True):
                g,msg = eng.step(); st.session_state.step += 1
                st.info(msg); st.rerun()
        with qc2:
            if st.button('⏩ Step Token (5×)', key='p5step', use_container_width=True):
                for _ in range(5): eng.step(); st.session_state.step += 1
                st.rerun()

# ═════════════════ TAB 4: REPLICATED LOGS ═════════════════════
with t4:
    st.markdown('### 📋 Replicated Request Logs at Every MSS')
    st.info('Each MSS maintains a **replicated log** of all requests it knows about — '
            'both from its own MHs and from broadcasts received from other MSSs.')

    for mss in ring.nodes:
        tok = ' 🔑' if mss.has_token else ''
        with st.expander(f'MSS_{mss.id}{tok} — {len(mss.rep_log)} log entries',
                         expanded=bool(mss.rep_log)):
            if mss.rep_log:
                df = pd.DataFrame([r.row() for r in mss.rep_log])
                # Color code by status
                st.dataframe(df, use_container_width=True, hide_index=True)

                st.caption(f'Local requests: '
                           f'{sum(1 for r in mss.rep_log if r.source_mss==mss.id)} | '
                           f'Replicated from others: '
                           f'{sum(1 for r in mss.rep_log if r.source_mss!=mss.id)}')
            else:
                st.caption('📭 No log entries yet — send requests from the sidebar!')

    st.markdown('---')
    st.markdown('#### Replication Summary')
    summary = []
    for mss in ring.nodes:
        local = sum(1 for r in mss.rep_log if r.source_mss == mss.id)
        remote = sum(1 for r in mss.rep_log if r.source_mss != mss.id)
        summary.append({"MSS": f"MSS_{mss.id}", "Local Entries": local,
                         "Replicated Entries": remote, "Total": local+remote})
    st.dataframe(pd.DataFrame(summary), use_container_width=True, hide_index=True)

# ═════════════════ TAB 5: QUEUE STATES ════════════════════════
with t5:
    st.markdown('### 📊 Queue States After Service')
    st.info('Shows the **current queue** at each MSS — pending requests waiting '
            'for the token, and completed/granted requests.')

    st.plotly_chart(static_ring(ring), use_container_width=True)

    cols = st.columns(ring.n)
    for i, mss in enumerate(ring.nodes):
        with cols[i]:
            tok = ' 🔑' if mss.has_token else ''
            st.markdown(f'**MSS_{mss.id}{tok}**')

            # Local queue
            pending = [r for r in mss.local_q if r.status == "PENDING"]
            granted = [r for r in mss.global_q if r.source_mss == mss.id and r.status == "GRANTED"]
            completed = [r for r in mss.global_q if r.source_mss == mss.id and r.status == "COMPLETED"]

            if pending:
                st.error(f'⏳ {len(pending)} pending')
                st.dataframe(pd.DataFrame([{"MH":r.mh_id,"P":r.priority,"T":r.timestamp}
                                           for r in pending]),hide_index=True)
            else:
                st.success('✅ No pending')

            if granted:
                st.warning(f'🟢 {len(granted)} granted')
            if completed:
                st.info(f'✔️ {len(completed)} completed')

    st.markdown('---')
    st.markdown('#### MSS Statistics')
    st.dataframe(pd.DataFrame([mss.stats() for mss in ring.nodes]),
                 use_container_width=True, hide_index=True)

    st.markdown('---')
    st.markdown('#### Engine Event Log')
    if eng.log:
        for ev in eng.log[-15:]:
            st.text(ev)
    else:
        st.caption('No events yet — use Step/×5 in the sidebar.')

st.markdown('---')
st.caption('Token-Ring Mutual Exclusion with Replication | MSS-MH Architecture')
