import React, { useState, useEffect, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  ShieldCheck, ShieldAlert, Lock,
  Activity, Ghost, Terminal as TerminalIcon, RotateCcw, Play, Zap,
  Server, Monitor, ArrowDown
} from 'lucide-react';
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid } from 'recharts';

const API_BASE = 'http://127.0.0.1:8000/api';

const App = () => {
  const [logs, setLogs] = useState([]);
  const [handshake, setHandshake] = useState(null);
  const [threatLevel, setThreatLevel] = useState("LOW");
  const [epsilon, setEpsilon] = useState(10.0);
  const [currentRound, setCurrentRound] = useState(0);
  const [maxRounds] = useState(10);
  const [accuracyHistory, setAccuracyHistory] = useState([]);
  const [trainingComplete, setTrainingComplete] = useState(false);
  const [isTraining, setIsTraining] = useState(false);
  const [encryptedSnippet, setEncryptedSnippet] = useState(null);
  const [globalWeightsSample, setGlobalWeightsSample] = useState(null);
  const [clientStatuses, setClientStatuses] = useState([
    { client_id: 'client_1', status: 'IDLE', num_samples: 0, local_accuracy: 0, privacy_spent: 0 },
    { client_id: 'client_2', status: 'IDLE', num_samples: 0, local_accuracy: 0, privacy_spent: 0 },
    { client_id: 'client_3', status: 'IDLE', num_samples: 0, local_accuracy: 0, privacy_spent: 0 },
  ]);
  const [serverAggregating, setServerAggregating] = useState(false);
  const [apiReady, setApiReady] = useState(false);
  const [attackActive, setAttackActive] = useState(null); // 'mitm' | 'quantum' | null
  const [attackPhase, setAttackPhase] = useState(''); // 'detecting', 'intercepted', 'blocked', 'breach', 'rotating', 'secured'

  const scrollRef = useRef(null);

  useEffect(() => {
    const interval = setInterval(fetchStatus, 3000);
    return () => clearInterval(interval);
  }, []);

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [logs]);

  const addLog = (msg, type = 'info') => {
    const timestamp = new Date().toLocaleTimeString();
    setLogs(prev => [...prev.slice(-39), { timestamp, msg, type }]);
  };

  const fetchStatus = async () => {
    try {
      const res = await fetch(`${API_BASE}/status`);
      const data = await res.json();
      setThreatLevel(data.threat_level);
      setCurrentRound(data.round);
      setTrainingComplete(data.training_complete);
      if (data.accuracy_history?.length > 0) setAccuracyHistory(data.accuracy_history);
      if (data.status === 'online') setApiReady(true);
    } catch (e) { /* backend not up */ }
  };

  const runHandshake = async () => {
    addLog("Initiating Kyber-768 Handshake...", 'info');
    try {
      const res = await fetch(`${API_BASE}/handshake`);
      const data = await res.json();
      if (data.status.includes("SECURE")) {
        setHandshake(data);
        addLog(`Quantum Channel Secured: ${data.status}`, 'success');
        addLog(`Session Key: ${data.session_key.substring(0, 16)}...`, 'info');
        setClientStatuses(prev => prev.map(c => ({ ...c, status: 'READY' })));
      }
    } catch (e) {
      addLog("Handshake Failed! Backend Unreachable.", 'error');
    }
  };

  const trainOneRound = async () => {
    if (!handshake) { addLog("No Secure Channel! Initiate Handshake first.", 'error'); return; }
    if (trainingComplete) { addLog("Training complete. Reset to train again.", 'warn'); return; }

    setIsTraining(true);
    const nextRound = currentRound + 1;
    addLog(`--- Round ${nextRound}/${maxRounds} ---`, 'info');

    // Animate: clients training
    setClientStatuses(prev => prev.map(c => ({ ...c, status: 'TRAINING' })));
    await sleep(400);

    // Animate: clients sending
    setClientStatuses(prev => prev.map(c => ({ ...c, status: 'ENCRYPTING' })));
    await sleep(400);

    try {
      const res = await fetch(`${API_BASE}/train/round`);
      const data = await res.json();

      // Animate: clients sent, server aggregating
      setClientStatuses(data.clients.map(c => ({ ...c, status: 'SENT' })));
      setServerAggregating(true);
      await sleep(600);

      // Update state
      setCurrentRound(data.round);
      setAccuracyHistory(data.accuracy_history);
      setTrainingComplete(data.training_complete);
      setEncryptedSnippet(data.encrypted_snippet);
      setGlobalWeightsSample(data.global_weights_sample);
      setServerAggregating(false);

      // Update client statuses from API response
      setClientStatuses(data.clients.map(c => ({
        ...c,
        status: data.training_complete ? 'COMPLETE' : 'READY',
      })));

      // Logs
      data.clients.forEach(c => {
        addLog(`[${c.client_id}] ${c.num_samples} samples | Acc: ${(c.local_accuracy * 100).toFixed(1)}% | DP: ${c.privacy_spent}`, 'info');
      });
      addLog(`Server aggregated (Paillier HE, blindfolded).`, 'warn');
      addLog(`Round ${data.round} Global Accuracy: ${(data.accuracy * 100).toFixed(2)}%`, 'success');

      if (data.training_complete) addLog(`All ${maxRounds} rounds complete!`, 'success');
    } catch (e) {
      addLog("Training round failed.", 'error');
      setServerAggregating(false);
    } finally {
      setIsTraining(false);
    }
  };

  const trainAllRounds = async () => {
    if (!handshake) { addLog("No Secure Channel! Initiate Handshake first.", 'error'); return; }
    if (trainingComplete) { addLog("Training complete. Reset to train again.", 'warn'); return; }

    setIsTraining(true);
    addLog("Running all remaining rounds...", 'info');
    setClientStatuses(prev => prev.map(c => ({ ...c, status: 'TRAINING' })));

    try {
      const res = await fetch(`${API_BASE}/train/auto`);
      const data = await res.json();

      setAccuracyHistory(data.accuracy_history);
      setCurrentRound(data.total_rounds);
      setTrainingComplete(true);
      setClientStatuses(prev => prev.map(c => ({ ...c, status: 'COMPLETE' })));

      addLog(`${data.msg}. Final Accuracy: ${(data.final_accuracy * 100).toFixed(2)}%`, 'success');
    } catch (e) {
      addLog("Auto-training failed.", 'error');
    } finally {
      setIsTraining(false);
    }
  };

  const resetPipeline = async () => {
    try {
      await fetch(`${API_BASE}/reset`, { method: 'POST' });
      setHandshake(null);
      setAccuracyHistory([]);
      setCurrentRound(0);
      setTrainingComplete(false);
      setEncryptedSnippet(null);
      setGlobalWeightsSample(null);
      setClientStatuses([
        { client_id: 'client_1', status: 'IDLE', num_samples: 0, local_accuracy: 0, privacy_spent: 0 },
        { client_id: 'client_2', status: 'IDLE', num_samples: 0, local_accuracy: 0, privacy_spent: 0 },
        { client_id: 'client_3', status: 'IDLE', num_samples: 0, local_accuracy: 0, privacy_spent: 0 },
      ]);
      setLogs([]);
      addLog("Pipeline reset. Ready for new training.", 'success');
    } catch (e) {
      addLog("Reset failed.", 'error');
    }
  };

  const updatePrivacy = async (val) => {
    const newEps = parseFloat(val);
    setEpsilon(newEps);
    await fetch(`${API_BASE}/privacy/config`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ epsilon: newEps })
    });
    addLog(`Privacy epsilon = ${newEps} on all clients`, 'warn');
  };

  const triggerAttack = async (type) => {
    if (attackActive) return; // prevent double-trigger

    if (type === 'mitm') {
      // === MITM ATTACK SEQUENCE ===
      setAttackActive('mitm');
      setThreatLevel('HIGH');

      // Phase 1: Attacker detected
      setAttackPhase('detecting');
      addLog('[ALERT] ANOMALOUS PACKET DETECTED ON NETWORK...', 'error');
      setClientStatuses(prev => prev.map((c, i) =>
        i === 1 ? { ...c, status: 'COMPROMISED' } : c
      ));
      await sleep(1500);

      // Phase 2: Interception attempt
      setAttackPhase('intercepted');
      addLog('[ATTACK] MITM ATTACKER INTERCEPTING CLIENT_2 UPDATE...', 'error');
      addLog('[ATTACK] Attempting to inject poisoned model weights...', 'error');
      await sleep(2000);

      // Phase 3: Dilithium blocks it
      setAttackPhase('blocked');
      addLog('[DEFENSE] DILITHIUM SIGNATURE VERIFICATION FAILED!', 'warn');
      addLog('[DEFENSE] Tampered payload REJECTED. Attack neutralized.', 'success');
      addLog('[DEFENSE] All client signatures re-verified. Network SECURE.', 'success');
      setThreatLevel('LOW');
      setClientStatuses(prev => prev.map(c => ({
        ...c, status: c.status === 'COMPROMISED' ? 'BLOCKED' : c.status
      })));
      await sleep(2500);

      // Reset
      setAttackActive(null);
      setAttackPhase('');
      setClientStatuses(prev => prev.map(c => ({
        ...c, status: c.num_samples > 0 ? 'READY' : (handshake ? 'READY' : 'IDLE')
      })));

    } else if (type === 'quantum') {
      // === QUANTUM ATTACK SEQUENCE ===
      setAttackActive('quantum');
      setThreatLevel('CRITICAL');

      // Phase 1: Quantum threat detected
      setAttackPhase('breach');
      addLog('[CRITICAL] QUANTUM COMPUTING WAVEFRONT DETECTED!', 'error');
      addLog('[CRITICAL] Shor\'s algorithm targeting Kyber key exchange...', 'error');
      await sleep(2000);

      // Phase 2: Key rotation
      setAttackPhase('rotating');
      addLog('[DEFENSE] INITIATING EMERGENCY KEY ROTATION...', 'warn');
      addLog('[DEFENSE] Generating fresh ML-KEM-768 keypairs...', 'warn');
      addLog('[DEFENSE] Re-encapsulating session keys for all 3 clients...', 'warn');
      await sleep(2500);

      // Phase 3: Secured
      setAttackPhase('secured');
      addLog('[DEFENSE] All keys rotated successfully.', 'success');
      addLog('[DEFENSE] Post-quantum encryption intact. Attack DEFEATED.', 'success');
      setThreatLevel('LOW');
      await sleep(2000);

      // Reset
      setAttackActive(null);
      setAttackPhase('');
    }

    // Also call the backend
    try {
      await fetch(`${API_BASE}/attack/simulate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ attack_type: type })
      });
    } catch (e) { /* optional backend call */ }
  };

  return (
    <div className={`min-h-screen text-slate-200 font-sans selection:bg-cyber-cyan selection:text-cyber-dark p-4 transition-colors duration-500 ${attackActive === 'mitm' ? 'bg-red-950/40' :
      attackActive === 'quantum' ? 'bg-purple-950/40' :
        'bg-cyber-bg'
      }`}>

      {/* ATTACK OVERLAY */}
      <AnimatePresence>
        {attackActive && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 z-50 pointer-events-none flex items-center justify-center"
          >
            {/* Scanline effect */}
            <div className="absolute inset-0 bg-gradient-to-b from-transparent via-red-500/5 to-transparent animate-pulse" />

            {/* Alert Banner */}
            <motion.div
              initial={{ scale: 0.8, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              exit={{ scale: 0.8, opacity: 0 }}
              className={`px-12 py-6 rounded-xl border-2 backdrop-blur-xl ${attackPhase === 'blocked' || attackPhase === 'secured'
                ? 'border-cyber-green bg-green-950/80'
                : attackActive === 'quantum'
                  ? 'border-purple-500 bg-purple-950/80'
                  : 'border-red-500 bg-red-950/80'
                }`}
            >
              <div className={`text-3xl font-bold font-mono tracking-widest text-center ${attackPhase === 'blocked' || attackPhase === 'secured'
                ? 'text-cyber-green'
                : 'text-red-400 animate-pulse'
                }`}>
                {attackPhase === 'detecting' && 'INTRUSION DETECTED'}
                {attackPhase === 'intercepted' && 'MITM ATTACK IN PROGRESS'}
                {attackPhase === 'blocked' && 'ATTACK NEUTRALIZED'}
                {attackPhase === 'breach' && 'QUANTUM THREAT DETECTED'}
                {attackPhase === 'rotating' && 'ROTATING KEYS...'}
                {attackPhase === 'secured' && 'SYSTEM SECURED'}
              </div>
              <div className="text-center font-mono text-sm mt-2 text-slate-300">
                {attackPhase === 'detecting' && 'Dilithium signature verification in progress...'}
                {attackPhase === 'intercepted' && 'Poisoned weights detected in client_2 update'}
                {attackPhase === 'blocked' && 'ML-DSA-44 signature mismatch -- payload rejected'}
                {attackPhase === 'breach' && "Shor's algorithm targeting key exchange"}
                {attackPhase === 'rotating' && 'Generating fresh ML-KEM-768 keypairs for all clients'}
                {attackPhase === 'secured' && 'Post-quantum encryption intact'}
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* HEADER */}
      <header className="flex justify-between items-center mb-6 border-b border-slate-800 pb-3">
        <div className="flex items-center gap-3">
          <ShieldCheck className="w-9 h-9 text-cyber-cyan animate-pulse" />
          <div>
            <h1 className="text-2xl font-bold tracking-tighter text-white">
              QUANTUM<span className="text-cyber-cyan">SAFE</span>.AI
            </h1>
            <p className="text-[10px] text-slate-500 font-mono">FEDERATED LEARNING DASHBOARD // V3.0</p>
          </div>
        </div>
        <div className="flex gap-4 items-center">
          <StatusBadge label="ROUND" value={`${currentRound}/${maxRounds}`}
            color={trainingComplete ? 'text-cyber-green' : 'text-cyber-cyan'} />
          <StatusBadge label="THREAT" value={threatLevel}
            color={threatLevel === 'LOW' ? 'text-cyber-green' : 'text-cyber-red animate-pulse'} />
          <StatusBadge label="CHANNEL" value={handshake ? "KYBER-768" : "UNSECURED"}
            color={handshake ? 'text-cyber-cyan' : 'text-slate-500'} />
          <button onClick={resetPipeline} className="cyber-button flex items-center gap-1 text-xs py-1 px-3">
            <RotateCcw className="w-3 h-3" /> RESET
          </button>
        </div>
      </header>

      <main className="grid grid-cols-12 gap-4" style={{ height: 'calc(100vh - 110px)' }}>

        {/* LEFT COL: CONTROLS */}
        <div className="col-span-2 space-y-4 flex flex-col">
          {/* Handshake */}
          <div className="glass-panel space-y-3">
            <h2 className="text-cyber-cyan font-mono text-xs flex items-center gap-2">
              <Lock className="w-3 h-3" /> SECURE CHANNEL
            </h2>
            <div className="text-[10px] text-slate-400 font-mono break-all">
              {handshake ? (
                <><p>PK: {handshake.public_key}</p><p className="mt-1 text-cyber-green">ACTIVE</p></>
              ) : (
                <p>No session.</p>
              )}
            </div>
            <button onClick={runHandshake} disabled={handshake} className="cyber-button w-full text-xs py-1.5">
              {handshake ? "ACTIVE" : "HANDSHAKE"}
            </button>
          </div>

          {/* Privacy */}
          <div className="glass-panel space-y-3">
            <h2 className="text-cyber-cyan font-mono text-xs flex items-center gap-2">
              <Ghost className="w-3 h-3" /> PRIVACY (LDP)
            </h2>
            <div className="space-y-1">
              <div className="flex justify-between text-[10px] font-mono">
                <span>PRIVATE</span><span>ACCURATE</span>
              </div>
              <input type="range" min="0.1" max="20.0" step="0.1" value={epsilon}
                onChange={(e) => updatePrivacy(e.target.value)}
                className="w-full accent-cyber-cyan h-1 bg-slate-700 rounded-lg appearance-none cursor-pointer" />
              <div className="text-center font-mono text-lg text-white">
                e = {epsilon.toFixed(1)}
              </div>
            </div>
          </div>

          {/* Attack Sim */}
          <div className="glass-panel space-y-3 flex-1">
            <h2 className="text-cyber-red font-mono text-xs flex items-center gap-2">
              <ShieldAlert className="w-3 h-3" /> THREATS
            </h2>
            <div className="grid gap-2">
              <button onClick={() => triggerAttack('mitm')} disabled={!!attackActive}
                className="cyber-button border-red-500/50 text-red-400 hover:bg-red-500/10 text-xs py-1.5 disabled:opacity-30">
                {attackActive === 'mitm' ? 'ATTACKING...' : 'MITM INTERCEPT'}
              </button>
              <button onClick={() => triggerAttack('quantum')} disabled={!!attackActive}
                className="cyber-button border-red-500/50 text-red-400 hover:bg-red-500/10 text-xs py-1.5 disabled:opacity-30">
                {attackActive === 'quantum' ? 'ATTACKING...' : 'QUANTUM ATTACK'}
              </button>
            </div>
          </div>
        </div>

        {/* MIDDLE COL: FEDERATION + CHART */}
        <div className="col-span-7 space-y-4 flex flex-col">

          {/* FEDERATION TOPOLOGY */}
          <div className="glass-panel">
            <h2 className="text-cyber-cyan font-mono text-xs mb-3 flex items-center gap-2">
              <Activity className="w-3 h-3" /> FEDERATED NETWORK
            </h2>

            <div className="flex items-start justify-center gap-6">
              {/* 3 CLIENT NODES */}
              {clientStatuses.map((client, idx) => (
                <ClientNode key={client.client_id} client={client} index={idx} attackActive={attackActive} attackPhase={attackPhase} />
              ))}
            </div>

            {/* ARROWS: Clients → Server */}
            <div className="flex justify-center my-2">
              <div className="flex gap-16">
                {clientStatuses.map((c, i) => (
                  <motion.div key={i}
                    animate={c.status === 'ENCRYPTING' || c.status === 'SENT' ? {
                      opacity: [0.3, 1, 0.3], y: [0, 4, 0]
                    } : {}}
                    transition={{ duration: 0.8, repeat: c.status === 'ENCRYPTING' ? Infinity : 0 }}
                    className="flex flex-col items-center"
                  >
                    <ArrowDown className={`w-4 h-4 ${c.status === 'ENCRYPTING' ? 'text-yellow-400' :
                      c.status === 'SENT' ? 'text-cyber-green' :
                        'text-slate-700'
                      }`} />
                    <span className={`text-[8px] font-mono mt-0.5 ${c.status === 'ENCRYPTING' ? 'text-yellow-400' :
                      c.status === 'SENT' ? 'text-cyber-green' :
                        'text-slate-700'
                      }`}>
                      {c.status === 'ENCRYPTING' ? 'HE+PQC' :
                        c.status === 'SENT' ? 'SENT' : 'IDLE'}
                    </span>
                  </motion.div>
                ))}
              </div>
            </div>

            {/* SERVER NODE */}
            <div className="flex justify-center">
              <ServerNode aggregating={serverAggregating}
                accuracy={accuracyHistory.length > 0 ? accuracyHistory[accuracyHistory.length - 1].accuracy : null}
                round={currentRound} attackActive={attackActive} attackPhase={attackPhase} />
            </div>
          </div>

          {/* ACCURACY CHART */}
          <div className="glass-panel flex-1 relative overflow-hidden">
            <h2 className="text-cyber-cyan font-mono text-xs mb-2 flex items-center gap-2">
              <Activity className="w-3 h-3" /> GLOBAL MODEL ACCURACY
            </h2>

            {accuracyHistory.length > 0 ? (
              <ResponsiveContainer width="100%" height="85%">
                <LineChart data={accuracyHistory}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                  <XAxis dataKey="round" stroke="#64748b" fontSize={10} tickFormatter={(v) => `R${v}`} />
                  <YAxis stroke="#64748b" fontSize={10} domain={[0.5, 0.85]}
                    tickFormatter={(v) => `${(v * 100).toFixed(0)}%`} />
                  <Tooltip
                    contentStyle={{ backgroundColor: '#0f172a', border: '1px solid #22d3ee', borderRadius: '8px', fontSize: '11px' }}
                    labelFormatter={(v) => `Round ${v}`}
                    formatter={(v) => [`${(v * 100).toFixed(2)}%`, 'Accuracy']} />
                  <Line type="monotone" dataKey="accuracy" stroke="#22d3ee" strokeWidth={2}
                    dot={{ r: 4, fill: '#22d3ee', stroke: '#0f172a', strokeWidth: 2 }}
                    activeDot={{ r: 6, fill: '#22d3ee' }} />
                </LineChart>
              </ResponsiveContainer>
            ) : (
              <div className="h-full flex items-center justify-center text-slate-600 font-mono text-xs">
                {apiReady ? "Initiate Handshake, then Train." : "Connecting to backend..."}
              </div>
            )}
          </div>

          {/* ACTION BAR */}
          <div className="glass-panel flex gap-3">
            <button onClick={trainOneRound} disabled={isTraining || trainingComplete}
              className="flex-1 py-3 bg-cyber-cyan/10 border border-cyber-cyan text-cyber-cyan font-bold rounded
                hover:bg-cyber-cyan hover:text-slate-900 transition-all tracking-widest text-sm
                disabled:opacity-40 disabled:cursor-not-allowed flex items-center justify-center gap-2">
              <Play className="w-4 h-4" />
              {isTraining ? "TRAINING..." : "RUN 1 ROUND"}
            </button>
            <button onClick={trainAllRounds} disabled={isTraining || trainingComplete}
              className="flex-1 py-3 bg-cyber-green/10 border border-cyber-green text-cyber-green font-bold rounded
                hover:bg-cyber-green hover:text-slate-900 transition-all tracking-widest text-sm
                disabled:opacity-40 disabled:cursor-not-allowed flex items-center justify-center gap-2">
              <Zap className="w-4 h-4" />
              {trainingComplete ? "COMPLETE" : "RUN ALL 10 ROUNDS"}
            </button>
          </div>
        </div>

        {/* RIGHT COL: LOGS */}
        <div className="col-span-3 glass-panel flex flex-col">
          <h2 className="text-slate-400 font-mono text-xs mb-2 flex items-center gap-2">
            <TerminalIcon className="w-3 h-3" /> SYSTEM LOGS
          </h2>
          <div ref={scrollRef} className="flex-1 overflow-y-auto font-mono text-[10px] space-y-1.5 pr-1">
            {logs.length === 0 && <span className="text-slate-600">Waiting for system events...</span>}
            {logs.map((log, i) => (
              <div key={i} className={
                log.type === 'error' ? 'text-cyber-red' :
                  log.type === 'warn' ? 'text-yellow-400' :
                    log.type === 'success' ? 'text-cyber-green' : 'text-cyber-cyan'
              }>
                <span className="opacity-50">[{log.timestamp}]</span> {log.msg}
              </div>
            ))}
          </div>
        </div>

      </main>
    </div>
  );
};


// ─── Client Node ──────────────────────────────────────────────
const ClientNode = ({ client, index, attackActive, attackPhase }) => {
  const labels = ['Hospital A', 'Bank B', 'Agency C'];

  const isCompromised = client.status === 'COMPROMISED';
  const isBlocked = client.status === 'BLOCKED';

  const colors = {
    IDLE: 'border-slate-700 text-slate-500',
    READY: 'border-cyber-cyan/50 text-cyber-cyan',
    TRAINING: 'border-yellow-500/80 text-yellow-400',
    ENCRYPTING: 'border-yellow-500/80 text-yellow-400',
    SENT: 'border-cyber-green/80 text-cyber-green',
    COMPLETE: 'border-cyber-green text-cyber-green',
    COMPROMISED: 'border-red-500 text-red-400',
    BLOCKED: 'border-yellow-500 text-yellow-400',
  };
  const bgColors = {
    IDLE: 'bg-slate-900/60',
    READY: 'bg-cyber-cyan/5',
    TRAINING: 'bg-yellow-500/5',
    ENCRYPTING: 'bg-yellow-500/5',
    SENT: 'bg-cyber-green/5',
    COMPLETE: 'bg-cyber-green/5',
    COMPROMISED: 'bg-red-500/15',
    BLOCKED: 'bg-yellow-500/10',
  };

  return (
    <motion.div
      animate={
        isCompromised ? { scale: [1, 1.06, 1], x: [-2, 2, -2, 0] } :
          client.status === 'TRAINING' ? { scale: [1, 1.03, 1] } : {}
      }
      transition={{
        duration: isCompromised ? 0.3 : 0.6,
        repeat: isCompromised || client.status === 'TRAINING' ? Infinity : 0
      }}
      className={`relative rounded-lg border px-4 py-3 w-44 ${colors[client.status] || colors.IDLE} ${bgColors[client.status] || bgColors.IDLE}`}
    >
      <div className="flex items-center gap-2 mb-2">
        <Monitor className={`w-4 h-4 ${isCompromised ? 'text-red-400' : ''}`} />
        <div>
          <div className="text-xs font-bold font-mono">{labels[index]}</div>
          <div className="text-[9px] font-mono opacity-60">{client.client_id}</div>
        </div>
      </div>

      <div className="space-y-1 text-[9px] font-mono">
        <div className="flex justify-between">
          <span className="opacity-60">STATUS</span>
          <span className={`font-bold ${isCompromised ? 'animate-pulse text-red-400' :
              isBlocked ? 'text-yellow-400' :
                client.status === 'TRAINING' ? 'animate-pulse text-yellow-400' :
                  client.status === 'COMPLETE' ? 'text-cyber-green' : ''
            }`}>{client.status}</span>
        </div>
        {client.num_samples > 0 && (
          <>
            <div className="flex justify-between">
              <span className="opacity-60">SAMPLES</span>
              <span>{client.num_samples.toLocaleString()}</span>
            </div>
            <div className="flex justify-between">
              <span className="opacity-60">LOCAL ACC</span>
              <span>{(client.local_accuracy * 100).toFixed(1)}%</span>
            </div>
            <div className="flex justify-between">
              <span className="opacity-60">DP SPENT</span>
              <span>{client.privacy_spent}</span>
            </div>
          </>
        )}
      </div>

      {/* Pulse indicators */}
      {(client.status === 'TRAINING' || client.status === 'ENCRYPTING') && (
        <div className="absolute -top-1 -right-1 w-2.5 h-2.5 rounded-full bg-yellow-400 animate-ping" />
      )}
      {client.status === 'SENT' && (
        <div className="absolute -top-1 -right-1 w-2.5 h-2.5 rounded-full bg-cyber-green" />
      )}
      {isCompromised && (
        <div className="absolute -top-1 -right-1 w-3 h-3 rounded-full bg-red-500 animate-ping" />
      )}
      {isBlocked && (
        <div className="absolute -top-1 -right-1 w-3 h-3 rounded-full bg-yellow-400" />
      )}
    </motion.div>
  );
};


// ─── Server Node ──────────────────────────────────────────────
const ServerNode = ({ aggregating, accuracy, round, attackActive, attackPhase }) => {
  const isUnderAttack = attackActive === 'quantum' && (attackPhase === 'breach' || attackPhase === 'rotating');
  const isSecured = attackPhase === 'secured' || attackPhase === 'blocked';

  return (
    <motion.div
      animate={
        isUnderAttack ? { scale: [1, 1.05, 1], boxShadow: ['0 0 0px #ef4444', '0 0 25px #ef4444', '0 0 0px #ef4444'] } :
          aggregating ? { scale: [1, 1.04, 1], boxShadow: ['0 0 0px #22d3ee', '0 0 20px #22d3ee', '0 0 0px #22d3ee'] } :
            isSecured ? { boxShadow: ['0 0 0px #22c55e', '0 0 15px #22c55e', '0 0 0px #22c55e'] } : {}
      }
      transition={{ duration: 0.6, repeat: isUnderAttack || aggregating ? Infinity : isSecured ? 2 : 0 }}
      className={`rounded-lg border px-6 py-3 w-72 text-center ${isUnderAttack ? 'border-red-500 bg-red-500/10' :
          isSecured ? 'border-cyber-green bg-cyber-green/10' :
            aggregating ? 'border-cyan-400 bg-cyan-500/10' :
              accuracy !== null ? 'border-cyber-green/60 bg-cyber-green/5' :
                'border-slate-700 bg-slate-900/60'
        }`}
    >
      <div className="flex items-center justify-center gap-2 mb-1">
        <Server className={`w-5 h-5 ${isUnderAttack ? 'text-red-400 animate-pulse' :
            aggregating ? 'text-cyber-cyan animate-spin' : 'text-slate-400'
          }`} />
        <span className="text-sm font-bold font-mono text-white">AGGREGATION SERVER</span>
      </div>
      <div className="text-[9px] font-mono text-slate-400 mb-1">
        Paillier HE (Blindfolded) + Dilithium Auth
      </div>
      {isUnderAttack && (
        <div className="text-xs font-mono text-red-400 animate-pulse font-bold">
          {attackPhase === 'breach' ? 'QUANTUM BREACH ATTEMPT!' : 'ROTATING ALL KEYS...'}
        </div>
      )}
      {isSecured && !aggregating && (
        <div className="text-xs font-mono text-cyber-green font-bold">
          DEFENSE ACTIVE -- SYSTEM SECURED
        </div>
      )}
      {aggregating && (
        <div className="text-xs font-mono text-cyber-cyan animate-pulse">
          AGGREGATING ENCRYPTED WEIGHTS...
        </div>
      )}
      {accuracy !== null && !aggregating && !isUnderAttack && !isSecured && (
        <div className="flex justify-center gap-4 text-[10px] font-mono mt-1">
          <span className="text-slate-400">ROUND: <span className="text-white">{round}</span></span>
          <span className="text-slate-400">GLOBAL ACC: <span className="text-cyber-green font-bold">{(accuracy * 100).toFixed(2)}%</span></span>
        </div>
      )}
    </motion.div>
  );
};


// ─── Status Badge ─────────────────────────────────────────────
const StatusBadge = ({ label, value, color }) => (
  <div className="text-right">
    <div className="text-[9px] text-slate-500 font-bold tracking-widest">{label}</div>
    <div className={`text-xs font-mono font-bold ${color}`}>{value}</div>
  </div>
);


// ─── Helpers ──────────────────────────────────────────────────
const sleep = (ms) => new Promise(r => setTimeout(r, ms));

export default App;
