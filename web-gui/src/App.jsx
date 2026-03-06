import React, { useState, useEffect, useRef } from 'react';
import { motion } from 'framer-motion';
import {
  ShieldCheck, ShieldAlert, Lock,
  Activity, Ghost, Terminal as TerminalIcon, RotateCcw, Play, Zap
} from 'lucide-react';
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid } from 'recharts';

// --- API Client ---
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
  const [apiReady, setApiReady] = useState(false);

  const scrollRef = useRef(null);

  // --- Effects ---
  useEffect(() => {
    const interval = setInterval(fetchStatus, 3000);
    return () => clearInterval(interval);
  }, []);

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [logs]);

  // --- Actions ---
  const addLog = (msg, type = 'info') => {
    const timestamp = new Date().toLocaleTimeString();
    setLogs(prev => [...prev.slice(-29), { timestamp, msg, type }]);
  };

  const fetchStatus = async () => {
    try {
      const res = await fetch(`${API_BASE}/status`);
      const data = await res.json();
      setThreatLevel(data.threat_level);
      setCurrentRound(data.round);
      setTrainingComplete(data.training_complete);
      if (data.accuracy_history && data.accuracy_history.length > 0) {
        setAccuracyHistory(data.accuracy_history);
      }
      if (data.status === 'online') setApiReady(true);
    } catch (e) {
      // API not up yet
    }
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
      }
    } catch (e) {
      addLog("Handshake Failed! Backend Unreachable.", 'error');
    }
  };

  const trainOneRound = async () => {
    if (!handshake) {
      addLog("Cannot train: No Secure Channel! Initiate Handshake first.", 'error');
      return;
    }
    if (trainingComplete) {
      addLog("Training already complete (10/10 rounds). Reset to train again.", 'warn');
      return;
    }

    setIsTraining(true);
    addLog(`Starting Round ${currentRound + 1}/${maxRounds}...`, 'info');

    try {
      const res = await fetch(`${API_BASE}/train/round`);
      const data = await res.json();

      setCurrentRound(data.round);
      setAccuracyHistory(data.accuracy_history);
      setTrainingComplete(data.training_complete);
      setEncryptedSnippet(data.encrypted_snippet);
      setGlobalWeightsSample(data.global_weights_sample);

      // Log each client
      data.clients.forEach(c => {
        addLog(`[${c.client_id}] Trained on ${c.num_samples} samples (DP: ${c.dp_noise_applied})`, 'info');
      });
      addLog(`Server aggregated blindfolded (Paillier HE).`, 'warn');
      addLog(`Round ${data.round} Accuracy: ${(data.accuracy * 100).toFixed(2)}%`, 'success');

      if (data.training_complete) {
        addLog(`[COMPLETE] All ${maxRounds} rounds finished!`, 'success');
      }
    } catch (e) {
      addLog("Training round failed. Is the backend running?", 'error');
    } finally {
      setIsTraining(false);
    }
  };

  const trainAllRounds = async () => {
    if (!handshake) {
      addLog("Cannot train: No Secure Channel! Initiate Handshake first.", 'error');
      return;
    }
    if (trainingComplete) {
      addLog("Training already complete. Reset to train again.", 'warn');
      return;
    }

    setIsTraining(true);
    addLog(`Running all remaining rounds...`, 'info');

    try {
      const res = await fetch(`${API_BASE}/train/auto`);
      const data = await res.json();

      setAccuracyHistory(data.accuracy_history);
      setCurrentRound(data.total_rounds);
      setTrainingComplete(true);

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
      setLogs([]);
      addLog("Pipeline reset. Ready for new training session.", 'success');
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
    addLog(`Privacy Budget Updated: epsilon = ${newEps} on all clients`, 'warn');
  };

  const triggerAttack = async (type) => {
    addLog(`[ALERT] SIMULATING ${type.toUpperCase()} ATTACK...`, 'error');
    const res = await fetch(`${API_BASE}/attack/simulate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ attack_type: type })
    });
    const data = await res.json();
    if (data.success) {
      addLog(`${data.msg} - ${data.action}`, 'success');
    } else {
      addLog(`${data.msg} - ${data.error}`, 'error');
    }
  };

  return (
    <div className="min-h-screen bg-cyber-bg text-slate-200 font-sans selection:bg-cyber-cyan selection:text-cyber-dark p-6">

      {/* HEADER */}
      <header className="flex justify-between items-center mb-8 border-b border-slate-800 pb-4">
        <div className="flex items-center gap-3">
          <ShieldCheck className="w-10 h-10 text-cyber-cyan animate-pulse" />
          <div>
            <h1 className="text-3xl font-bold tracking-tighter text-white">
              QUANTUM<span className="text-cyber-cyan">SAFE</span>.AI
            </h1>
            <p className="text-xs text-slate-500 font-mono">LIVE TRAINING DASHBOARD // V3.0</p>
          </div>
        </div>

        <div className="flex gap-4 items-center">
          <StatusBadge label="ROUND" value={`${currentRound}/${maxRounds}`}
            color={trainingComplete ? 'text-cyber-green' : 'text-cyber-cyan'} />
          <StatusBadge label="THREAT LEVEL" value={threatLevel}
            color={threatLevel === 'LOW' ? 'text-cyber-green' : 'text-cyber-red animate-pulse'} />
          <StatusBadge label="CHANNEL" value={handshake ? "ENCRYPTED (KYBER)" : "UNSECURED"}
            color={handshake ? 'text-cyber-cyan' : 'text-slate-500'} />
          <button onClick={resetPipeline} className="cyber-button flex items-center gap-2 text-xs">
            <RotateCcw className="w-3 h-3" /> RESET
          </button>
        </div>
      </header>

      <main className="grid grid-cols-12 gap-6 h-[80vh]">

        {/* LEFT COL: CONTROLS */}
        <div className="col-span-3 space-y-6 flex flex-col">
          {/* Handshake Panel */}
          <div className="glass-panel space-y-4">
            <h2 className="text-cyber-cyan font-mono text-sm flex items-center gap-2">
              <Lock className="w-4 h-4" /> SECURE CHANNEL
            </h2>
            <div className="text-xs text-slate-400 font-mono break-all">
              {handshake ? (
                <>
                  <p>PK: {handshake.public_key}</p>
                  <p className="mt-2 text-cyber-green">SESSION ACTIVE</p>
                </>
              ) : (
                <p>No active session key.</p>
              )}
            </div>
            <button onClick={runHandshake} disabled={handshake}
              className="cyber-button w-full">
              {handshake ? "SESSION ACTIVE" : "INITIATE HANDSHAKE"}
            </button>
          </div>

          {/* Privacy Tuner */}
          <div className="glass-panel space-y-4">
            <h2 className="text-cyber-cyan font-mono text-sm flex items-center gap-2">
              <Ghost className="w-4 h-4" /> PRIVACY TUNER (LDP)
            </h2>

            <div className="space-y-2">
              <div className="flex justify-between text-xs font-mono">
                <span>PRIVACY</span>
                <span>ACCURACY</span>
              </div>
              <input
                type="range" min="0.1" max="20.0" step="0.1"
                value={epsilon} onChange={(e) => updatePrivacy(e.target.value)}
                className="w-full accent-cyber-cyan h-1 bg-slate-700 rounded-lg appearance-none cursor-pointer"
              />
              <div className="text-center font-mono text-xl text-white">
                epsilon = {epsilon.toFixed(1)}
              </div>
              <p className="text-xs text-slate-500 text-center">Lower epsilon = More Noise = Stronger Privacy</p>
            </div>
          </div>

          {/* Attack Sim */}
          <div className="glass-panel space-y-4 flex-1">
            <h2 className="text-cyber-red font-mono text-sm flex items-center gap-2">
              <ShieldAlert className="w-4 h-4" /> THREAT SIMULATOR
            </h2>
            <div className="grid grid-cols-1 gap-2">
              <button onClick={() => triggerAttack('mitm')}
                className="cyber-button border-red-500/50 text-red-400 hover:bg-red-500/10">
                MITM INTERCEPT
              </button>
              <button onClick={() => triggerAttack('quantum')}
                className="cyber-button border-red-500/50 text-red-400 hover:bg-red-500/10">
                QUANTUM DECRYPTION
              </button>
            </div>
          </div>
        </div>

        {/* MIDDLE COL: VISUALIZATION */}
        <div className="col-span-6 space-y-6 flex flex-col">

          {/* Accuracy Chart */}
          <div className="glass-panel flex-1 relative overflow-hidden">
            <h2 className="text-cyber-cyan font-mono text-sm mb-4 flex items-center gap-2">
              <Activity className="w-4 h-4" /> LIVE MODEL ACCURACY
            </h2>

            {accuracyHistory.length > 0 ? (
              <ResponsiveContainer width="100%" height={250}>
                <LineChart data={accuracyHistory}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                  <XAxis
                    dataKey="round"
                    stroke="#64748b"
                    fontSize={12}
                    tickFormatter={(v) => `R${v}`}
                  />
                  <YAxis
                    stroke="#64748b"
                    fontSize={12}
                    domain={[0.5, 0.85]}
                    tickFormatter={(v) => `${(v * 100).toFixed(0)}%`}
                  />
                  <Tooltip
                    contentStyle={{ backgroundColor: '#0f172a', border: '1px solid #22d3ee', borderRadius: '8px' }}
                    labelFormatter={(v) => `Round ${v}`}
                    formatter={(v) => [`${(v * 100).toFixed(2)}%`, 'Accuracy']}
                  />
                  <Line
                    type="monotone"
                    dataKey="accuracy"
                    stroke="#22d3ee"
                    strokeWidth={3}
                    dot={{ r: 5, fill: '#22d3ee', stroke: '#0f172a', strokeWidth: 2 }}
                    activeDot={{ r: 7, fill: '#22d3ee' }}
                  />
                </LineChart>
              </ResponsiveContainer>
            ) : (
              <div className="h-64 flex items-center justify-center text-slate-600 font-mono text-sm">
                {apiReady
                  ? "Initiate Handshake, then click Train to see real accuracy data."
                  : "Connecting to backend..."}
              </div>
            )}

            {/* HE Visualization Overlay */}
            {encryptedSnippet && (
              <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }}
                className="mt-2 bg-slate-950/90 border border-cyber-green/30 p-3 rounded-lg">
                <h3 className="text-cyber-green font-mono text-xs mb-2">ENCRYPTED MODEL STATE (PAILLIER HE)</h3>
                <div className="grid grid-cols-2 gap-4 text-xs font-mono">
                  <div>
                    <span className="text-slate-500">DECRYPTED WEIGHTS (sample):</span>
                    <div className="text-white mt-1">[{globalWeightsSample && globalWeightsSample.map(n => n.toFixed(4)).join(', ')}]</div>
                  </div>
                  <div>
                    <span className="text-slate-500">SERVER VIEW (ENCRYPTED):</span>
                    <div className="text-cyber-green mt-1 truncate">{encryptedSnippet}</div>
                  </div>
                </div>
              </motion.div>
            )}
          </div>

          {/* Action Bar */}
          <div className="glass-panel flex gap-4">
            <button onClick={trainOneRound} disabled={isTraining || trainingComplete}
              className="flex-1 py-4 bg-cyber-cyan/10 border border-cyber-cyan text-cyber-cyan font-bold rounded
                hover:bg-cyber-cyan hover:text-slate-900 transition-all tracking-widest
                disabled:opacity-40 disabled:cursor-not-allowed disabled:hover:bg-cyber-cyan/10 disabled:hover:text-cyber-cyan
                flex items-center justify-center gap-2">
              <Play className="w-5 h-5" />
              {isTraining ? "TRAINING..." : "RUN 1 ROUND"}
            </button>
            <button onClick={trainAllRounds} disabled={isTraining || trainingComplete}
              className="flex-1 py-4 bg-cyber-green/10 border border-cyber-green text-cyber-green font-bold rounded
                hover:bg-cyber-green hover:text-slate-900 transition-all tracking-widest
                disabled:opacity-40 disabled:cursor-not-allowed disabled:hover:bg-cyber-green/10 disabled:hover:text-cyber-green
                flex items-center justify-center gap-2">
              <Zap className="w-5 h-5" />
              {trainingComplete ? "COMPLETE" : "RUN ALL 10 ROUNDS"}
            </button>
          </div>
        </div>

        {/* RIGHT COL: LOGS */}
        <div className="col-span-3 glass-panel flex flex-col">
          <h2 className="text-slate-400 font-mono text-sm mb-2 flex items-center gap-2">
            <TerminalIcon className="w-4 h-4" /> SYSTEM LOGS
          </h2>
          <div ref={scrollRef} className="flex-1 overflow-y-auto font-mono text-xs space-y-2 pr-2">
            {logs.length === 0 && <span className="text-slate-600">Waiting for system events...</span>}
            {logs.map((log, i) => (
              <div key={i} className={`
                        ${log.type === 'error' ? 'text-cyber-red' :
                  log.type === 'warn' ? 'text-yellow-400' :
                    log.type === 'success' ? 'text-cyber-green' : 'text-cyber-cyan'}
                    `}>
                <span className="opacity-50">[{log.timestamp}]</span> {log.msg}
              </div>
            ))}
          </div>
        </div>

      </main>
    </div>
  );
};

const StatusBadge = ({ label, value, color }) => (
  <div className="text-right">
    <div className="text-[10px] text-slate-500 font-bold tracking-widest">{label}</div>
    <div className={`text-sm font-mono font-bold ${color}`}>{value}</div>
  </div>
);

export default App;
