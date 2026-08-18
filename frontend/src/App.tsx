import { FormEvent, useEffect, useMemo, useRef, useState } from "react";

const API = import.meta.env.VITE_API_URL || window.location.origin;
type Station = { id: string; name: string; enabled: boolean };
type Person = { id: string; display_name: string; credential_id: string | null; version: number; sample_count: number };
type Entry = { id: string; person_id: string | null; person_name: string | null; station_id: string; decision: string; confidence: number | null; created_at: string; operator: string };
type Track = { track_id: string; box: number[]; status: string; confidence?: number; person_id?: string; person_name?: string; prior_entry_at?: string };
type FrozenSelection = Track & { capture_id: string; expires_in_seconds: number };
type Session = { token: string; username: string; role: string };

function statusLabel(status: string) {
  return ({ recognized: "Recognized", reentry: "Re-entry", ambiguous: "Needs review", unknown: "Unknown", low_quality: "Face unclear", recognition_unavailable: "Recognition unavailable" } as Record<string, string>)[status] || status;
}

function App() {
  const [session, setSession] = useState<Session | null>(() => {
    const stored = localStorage.getItem("gatewatch-session");
    return stored ? JSON.parse(stored) : null;
  });
  return session ? <Console session={session} signOut={() => { localStorage.removeItem("gatewatch-session"); setSession(null); }} /> : <Login onSession={(next) => { localStorage.setItem("gatewatch-session", JSON.stringify(next)); setSession(next); }} />;
}

function Login({ onSession }: { onSession: (session: Session) => void }) {
  const [username, setUsername] = useState("admin");
  const [password, setPassword] = useState("change-me-now");
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  async function submit(event: FormEvent) {
    event.preventDefault(); setBusy(true); setError("");
    const response = await fetch(`${API}/api/auth/token`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ username, password }) });
    const data = await response.json(); setBusy(false);
    if (!response.ok) { setError(data.detail || "Sign-in failed"); return; }
    onSession({ token: data.token, username: data.username, role: data.role });
  }
  return <main className="login-shell"><section className="login-card"><div className="mark">GW</div><p className="eyebrow">Conference access console</p><h1>Every entry,<br/>accounted for.</h1><p className="muted">Sign in at your assigned gate. Recognition suggests; your judgment records the decision.</p><form onSubmit={submit}><label>Operator ID<input value={username} onChange={(event) => setUsername(event.target.value)} autoComplete="username" /></label><label>Password<input type="password" value={password} onChange={(event) => setPassword(event.target.value)} autoComplete="current-password" /></label>{error && <p className="error">{error}</p>}<button className="primary wide" disabled={busy}>{busy ? "Signing in…" : "Open gate console"}</button></form><p className="footnote">Local venue system · Authorized operators only</p></section></main>;
}

function Console({ session, signOut }: { session: Session; signOut: () => void }) {
  const headers = { Authorization: `Bearer ${session.token}`, "Content-Type": "application/json" };
  const [stations, setStations] = useState<Station[]>([]);
  const [stationId, setStationId] = useState("");
  const [tab, setTab] = useState<"gate" | "people" | "activity">("gate");
  const [people, setPeople] = useState<Person[]>([]);
  const [entries, setEntries] = useState<Entry[]>([]);
  const [notice, setNotice] = useState("Connect a camera to start live recognition.");
  const load = async () => {
    const [stationResult, peopleResult, entryResult] = await Promise.all([fetch(`${API}/api/stations`, { headers }), fetch(`${API}/api/people`, { headers }), fetch(`${API}/api/entries`, { headers })]);
    if (stationResult.ok) { const values = await stationResult.json(); setStations(values); setStationId((current) => current || values[0]?.id || ""); }
    if (peopleResult.ok) setPeople(await peopleResult.json());
    if (entryResult.ok) setEntries(await entryResult.json());
  };
  useEffect(() => { void load(); }, []);
  const station = stations.find((item) => item.id === stationId);
  return <main className="console-shell"><header className="topbar"><div className="brand"><span className="brand-mark">GW</span><span>GATEWATCH <em>·</em> ENTRY CONTROL</span></div><div className="top-meta"><span className="live-dot"></span><span>{station?.name || "No station selected"}</span><span className="operator">{session.username} · {session.role}</span><button className="text-button" onClick={signOut}>Sign out</button></div></header><aside className="rail"><button className={tab === "gate" ? "nav active" : "nav"} onClick={() => setTab("gate")}><span>01</span> Live gate</button><button className={tab === "people" ? "nav active" : "nav"} onClick={() => setTab("people")}><span>02</span> People</button><button className={tab === "activity" ? "nav active" : "nav"} onClick={() => setTab("activity")}><span>03</span> Activity</button><div className="rail-bottom"><label>Station<select value={stationId} onChange={(event) => setStationId(event.target.value)}>{stations.map((item) => <option key={item.id} value={item.id}>{item.name}</option>)}</select></label></div></aside><section className="workspace">{tab === "gate" && stationId && <LiveGate stationId={stationId} headers={headers} notice={notice} setNotice={setNotice} reload={load} />}{tab === "people" && <People people={people} headers={headers} reload={load} isAdmin={session.role === "administrator"} />}{tab === "activity" && <Activity entries={entries} />}</section><footer className="statusbar"><span>{notice}</span><span>{people.length} enrolled · {entries.length} recent decisions</span></footer></main>;
}

function LiveGate({ stationId, headers, notice, setNotice, reload }: { stationId: string; headers: Record<string, string>; notice: string; setNotice: (value: string) => void; reload: () => Promise<void> }) {
  const video = useRef<HTMLVideoElement>(null); const canvas = useRef<HTMLCanvasElement>(null); const socket = useRef<WebSocket | null>(null);
  const [tracks, setTracks] = useState<Track[]>([]); const [selected, setSelected] = useState<FrozenSelection | null>(null); const [streaming, setStreaming] = useState(false); const [enrollName, setEnrollName] = useState(""); const [credential, setCredential] = useState(""); const [busy, setBusy] = useState(false); const selectionRequest = useRef(0); const frameInFlight = useRef(false); const nextIdentityFrameAt = useRef(0); const lastFrameKind = useRef<"identity" | "tracking">("identity");
  const token = headers.Authorization.replace("Bearer ", "");
  useEffect(() => { return () => { socket.current?.close(); video.current?.srcObject && (video.current.srcObject as MediaStream).getTracks().forEach((track) => track.stop()); }; }, []);
  useEffect(() => { setTracks([]); setSelected(null); socket.current?.close(); if (streaming) void start(); }, [stationId]);
  async function start() {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "user", width: { ideal: 1280, max: 1920 }, height: { ideal: 720, max: 1080 } }, audio: false });
      if (!video.current) return; video.current.srcObject = stream; await video.current.play();
      const wsUrl = API.replace(/^http/, "ws") + `/api/live/${stationId}?token=${encodeURIComponent(token)}`;
      const ws = new WebSocket(wsUrl); socket.current = ws; frameInFlight.current = false; nextIdentityFrameAt.current = 0;
      let timer: number | undefined;
      const scheduleFrame = (delay = 75) => { window.clearTimeout(timer); timer = window.setTimeout(captureFrame, delay); };
      const captureFrame = () => {
        if (ws.readyState !== WebSocket.OPEN || !video.current || !canvas.current || video.current.videoWidth === 0) { if (ws.readyState === WebSocket.OPEN) scheduleFrame(); return; }
        if (frameInFlight.current) return;
        const kind: "identity" | "tracking" = performance.now() >= nextIdentityFrameAt.current ? "identity" : "tracking";
        const sourceWidth = video.current.videoWidth; const sourceHeight = video.current.videoHeight; const maxDimension = kind === "identity" ? 1280 : 640;
        const scale = Math.min(1, maxDimension / Math.max(sourceWidth, sourceHeight));
        canvas.current.width = Math.round(sourceWidth * scale); canvas.current.height = Math.round(sourceHeight * scale);
        canvas.current.getContext("2d")?.drawImage(video.current, 0, 0, canvas.current.width, canvas.current.height);
        const quality = kind === "identity" ? 0.9 : 0.68;
        canvas.current.toBlob((blob) => {
          if (!blob || ws.readyState !== WebSocket.OPEN) { if (ws.readyState === WebSocket.OPEN) scheduleFrame(); return; }
          frameInFlight.current = true; lastFrameKind.current = kind;
          if (kind === "identity") nextIdentityFrameAt.current = performance.now() + 1200;
          ws.send(new Blob([kind === "identity" ? "I" : "T", blob], { type: "application/octet-stream" }));
        }, "image/jpeg", quality);
      };
      ws.onopen = () => { setStreaming(true); setNotice("Live tracking is active. Identity refreshes every 1.2 seconds."); scheduleFrame(0); };
      ws.onmessage = (event) => { const message = JSON.parse(event.data); frameInFlight.current = false; if (message.type === "tracks") { setTracks(message.tracks); } if (message.type === "error") setNotice(message.detail); scheduleFrame(lastFrameKind.current === "identity" ? 20 : 75); };
      ws.onclose = () => { window.clearTimeout(timer); frameInFlight.current = false; setStreaming(false); };
    } catch { setNotice("Camera permission is required. Check browser permissions and try again."); setStreaming(false); }
  }
  function stop() { socket.current?.close(); const source = video.current?.srcObject as MediaStream | null; source?.getTracks().forEach((track) => track.stop()); if (video.current) video.current.srcObject = null; setStreaming(false); setTracks([]); setSelected(null); setNotice("Live recognition paused."); }
  async function selectFace(track: Track) {
    const requestId = ++selectionRequest.current;
    setNotice("Holding the selected face for enrollment...");
    const response = await fetch(API + "/api/enrollment-captures", { method: "POST", headers, body: JSON.stringify({ station_id: stationId, track_id: track.track_id }) });
    const data = await response.json();
    if (requestId !== selectionRequest.current) return;
    if (!response.ok) {
      setNotice(data.detail || "Could not hold this face. Select it again.");
      return;
    }
    setSelected({ ...track, capture_id: data.capture_id, expires_in_seconds: data.expires_in_seconds });
    setNotice("Face held for enrollment. Finish the details even if the face leaves the camera view.");
  }
  async function decide(decision: "confirmed" | "denied" | "override") { if (!selected) return; setBusy(true); const response = await fetch(`${API}/api/entries`, { method: "POST", headers, body: JSON.stringify({ station_id: stationId, track_id: selected.track_id, person_id: selected.person_id || null, decision, confidence: selected.confidence || null, reason: decision === "override" ? "Operator override" : null }) }); const data = await response.json(); setBusy(false); if (!response.ok) { setNotice(data.detail || "Could not record the decision."); return; } setNotice(`${statusLabel(decision)} recorded${data.person_name ? ` for ${data.person_name}` : ""}.`); setSelected(null); await reload(); }
  async function enroll(allowDistinct = false) { if (!selected || !enrollName.trim()) return; setBusy(true); const response = await fetch(`${API}/api/enrollments`, { method: "POST", headers, body: JSON.stringify({ station_id: stationId, capture_id: selected.capture_id, display_name: enrollName.trim(), credential_id: credential || null, allow_distinct: allowDistinct }) }); const data = await response.json(); setBusy(false); if (data.status === "duplicate_review") { setNotice(`Possible existing match: ${data.candidates.map((item: { display_name: string }) => item.display_name).join(", ")}. Confirm a distinct enrollment if appropriate.`); return; } if (!response.ok) { if (response.status === 409) setSelected(null); setNotice(data.detail || "Could not enroll this person."); return; } setNotice(`${data.person.display_name} enrolled successfully.`); setEnrollName(""); setCredential(""); setSelected(null); await reload(); }
  const dimensions = video.current && video.current.videoWidth ? { w: video.current.videoWidth, h: video.current.videoHeight } : { w: 1, h: 1 };
  return <div className="gate-layout"><section className="camera-panel"><div className="panel-heading"><div><p className="eyebrow">Live feed</p><h2>{streaming ? "Gate is watching" : "Camera is idle"}</h2></div><button className={streaming ? "button muted-button" : "button primary"} onClick={streaming ? stop : () => void start()}>{streaming ? "Pause camera" : "Start camera"}</button></div><div className="viewfinder"><video ref={video} muted playsInline/><canvas ref={canvas} hidden/>{tracks.map((track) => { const [x, y, width, height] = track.box; return <button key={track.track_id} className={`face-box ${track.status} ${selected?.track_id === track.track_id ? "selected" : ""}`} style={{ left: `${x * 100}%`, top: `${y * 100}%`, width: `${width * 100}%`, height: `${height * 100}%` }} onClick={() => void selectFace(track)}><span>{track.person_name || statusLabel(track.status)}</span></button>; })}{!streaming && <div className="camera-empty"><span className="camera-icon">◉</span><strong>Camera preview is off</strong><p>Start the camera when your station is ready.</p></div>}</div><div className="legend"><span className="recognized">Recognized</span><span className="reentry">Re-entry</span><span className="unknown">Unknown</span><span className="low_quality">Face unclear</span><span className="recognition_unavailable">Recognition unavailable</span></div></section><aside className="decision-panel">{selected ? <><p className="eyebrow">Selected face · {selected.track_id.slice(0, 8)}</p><h2>{selected.person_name || "Unidentified visitor"}</h2><p className="capture-note">Face held for enrollment · live detection can continue</p><div className={"result-band " + selected.status}><span>{statusLabel(selected.status)}</span>{selected.confidence && <b>{Math.round(selected.confidence * 100)}% match</b>}</div>{selected.prior_entry_at && <p className="prior-entry">Last entered <strong>{new Date(selected.prior_entry_at).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}</strong> · Re-entry allowed</p>}{selected.person_id ? <div className="actions"><button className="primary" disabled={busy} onClick={() => void decide("confirmed")}>Confirm entry</button><button className="button" disabled={busy} onClick={() => void decide("override")}>Override match</button><button className="danger-link" disabled={busy} onClick={() => void decide("denied")}>Deny entry</button></div> : <div className="enrollment"><p className="muted">Nothing is saved until you enroll this person.</p><label>Name<input placeholder="Full name" value={enrollName} onChange={(event) => setEnrollName(event.target.value)} /></label><label>Credential ID <small>optional</small><input placeholder="Badge or registration ID" value={credential} onChange={(event) => setCredential(event.target.value)} /></label><button className="primary wide" disabled={busy || !enrollName.trim()} onClick={() => void enroll()}>Enroll person</button><button className="button wide" disabled={busy || !enrollName.trim()} onClick={() => void enroll(true)}>Enroll as distinct person</button></div>}</> : <div className="decision-empty"><div className="scan-lines"></div><h2>Select a face</h2><p>Live tracks appear here. Review the recognition before recording any entry.</p></div>}</aside></div>;
}

function People({ people, headers, reload, isAdmin }: { people: Person[]; headers: Record<string, string>; reload: () => Promise<void>; isAdmin: boolean }) {
  const [query, setQuery] = useState(""); const filtered = useMemo(() => people.filter((person) => person.display_name.toLowerCase().includes(query.toLowerCase())), [people, query]);
  async function remove(person: Person) { if (!confirm(`Permanently delete ${person.display_name} and all saved samples?`)) return; await fetch(`${API}/api/people/${person.id}`, { method: "DELETE", headers }); await reload(); }
  return <section className="directory"><div className="panel-heading"><div><p className="eyebrow">Face library</p><h2>Enrolled people</h2></div><input className="search" placeholder="Search people" value={query} onChange={(event) => setQuery(event.target.value)} /></div><div className="people-grid">{filtered.map((person) => <article className="person-card" key={person.id}><div className="avatar">{person.display_name.split(" ").map((part) => part[0]).slice(0, 2).join("")}</div><div><h3>{person.display_name}</h3><p>{person.credential_id || "No credential ID"}</p><small>{person.sample_count} face sample{person.sample_count === 1 ? "" : "s"}</small></div>{isAdmin && <button className="icon-button" title="Delete person" onClick={() => void remove(person)}>×</button>}</article>)}</div>{filtered.length === 0 && <div className="empty-state">No enrolled people match this search.</div>}</section>;
}

function Activity({ entries }: { entries: Entry[] }) { return <section className="directory"><div className="panel-heading"><div><p className="eyebrow">Decision ledger</p><h2>Recent gate activity</h2></div></div><div className="ledger">{entries.map((entry) => <div className="ledger-row" key={entry.id}><span className={`ledger-status ${entry.decision}`}></span><div><strong>{entry.person_name || "Unidentified visitor"}</strong><p>{entry.decision} by {entry.operator}</p></div><time>{new Date(entry.created_at).toLocaleString()}</time></div>)}{entries.length === 0 && <div className="empty-state">No gate decisions have been recorded yet.</div>}</div></section>; }

export default App;