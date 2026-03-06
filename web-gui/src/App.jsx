import React, { useState, useEffect, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  ShieldCheck, ShieldAlert, Lock, Unlock,
  Activity, Server, Database, Ghost, Terminal as TerminalIcon
} from 'lucide-react';
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer } from 'recharts';

// --- API Client ---
const API_BASE = 'http://127.0.0.1:8000/api';

const App = () => {
  const [status, setStatus] = useState(null);
  const [logs, setLogs] = useState([]);
  const [handshake, setHandshake] = useState(null);
  const [heData, setHeData] = useState(null);
  const [privacyBudget, setPrivacyBudget] = useState(100);
  const [epsilon, setEpsilon] = useState(2.0);
  const [threatLevel, setThreatLevel] = useState("LOW");

  const scrollRef = useRef(null);

  // --- Effects ---
  useEffect(() => {
    // Poll status every 2 seconds
    const interval = setInterval(fetchStatus, 2000);
    return () => clearInterval(interval);
  }, []);

  useEffect(() => {
    // Auto-scroll logs
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [logs]);

  // --- Actions ---
  const addLog = (msg, type = 'info') => {
    const timestamp = new Date().toLocaleTimeString();
    setLogs(prev => [...prev.slice(-19), { timestamp, msg, type }]);
  };

  const fetchStatus = async () => {
    try {
      const res = await fetch(`${API_BASE}/status`);
      const data = await res.json();
      setThreatLevel(data.threat_level);
      setPrivacyBudget(data.privacy_budget);
    } catch (e) {
      // console.error(e);
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
      addLog("Handshake Failed! Network Unreachable.", 'error');
    }
  };

  const simulateHE = async () => {
    if (!handshake) {
      addLog("Cannot train: No Secure Channel!", 'error');
      return;
    }

    addLog("Starting Encrypted Training Round...", 'info');
    try {
      const res = await fetch(`${API_BASE}/he/simulate`);
      const data = await res.json();
      setHeData(data);
      addLog("Client Weights Encrypted (Paillier).", 'info');
      addLog("Server Aggregating Blindfolded...", 'warn');
      setTimeout(() => {
        addLog(`Decrypted Aggregate: [${data.decrypted_sum.map(n => n.toFixed(2)).join(', ')}]`, 'success');
      }, 800);
    } catch (e) {
      addLog("Training Failed.", 'error');
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
    addLog(`Privacy Budget Updated: ε = ${newEps}`, 'warn');
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
            <p className="text-xs text-slate-500 font-mono">SOC DASHBOARD // V2.0.4</p>
          </div>
        </div>

        <div className="flex gap-4">
          <StatusBadge label="THREAT LEVEL" value={threatLevel}
            color={threatLevel === 'LOW' ? 'text-cyber-green' : 'text-cyber-red animate-pulse'} />
          <StatusBadge label="CHANNEL" value={handshake ? "ENCRYPTED (KYBER)" : "UNSECURED"}
            color={handshake ? 'text-cyber-cyan' : 'text-slate-500'} />
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
              {handshake ? "RE-KEY CHANNEL" : "INITIATE HANDSHAKE"}
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
                type="range" min="0.1" max="10.0" step="0.1"
                value={epsilon} onChange={(e) => updatePrivacy(e.target.value)}
                className="w-full accent-cyber-cyan h-1 bg-slate-700 rounded-lg appearance-none cursor-pointer"
              />
              <div className="text-center font-mono text-xl text-white">
                ε = {epsilon.toFixed(1)}
              </div>
              <p className="text-xs text-slate-500 text-center">Lower ε = More Noise</p>
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

          {/* Main Graph (Placeholder for Real Data) */}
          <div className="glass-panel flex-1 relative overflow-hidden group">
            <div className="absolute inset-0 bg-[url('https://grainy-gradients.vercel.app/noise.svg')] opacity-10"></div>
            <h2 className="text-cyber-cyan font-mono text-sm mb-4 flex items-center gap-2">
              <Activity className="w-4 h-4" /> NETWORK TRAFFIC ANALYSIS
            </h2>

            <div className="h-64 w-full flex items-end gap-1 justify-center opacity-80">
              {/* Simulated Bars */}
              {Array.from({ length: 40 }).map((_, i) => (
                <div key={i}
                  style={{ height: `${20 + Math.random() * 60}%` }}
                  className="w-2 bg-cyber-cyan/30 rounded-t hover:bg-cyber-cyan transition-all"
                />
              ))}
            </div>

            {/* HE Visualization Overlay */}
            {heData && (
              <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }}
                className="absolute bottom-4 left-4 right-4 bg-slate-950/90 border border-cyber-green/30 p-4 rounded-lg">
                <h3 className="text-cyber-green font-mono text-xs mb-2">HOMOMORPHIC AGGREGATION RESULT</h3>
                <div className="grid grid-cols-2 gap-4 text-xs font-mono">
                  <div>
                    <span className="text-slate-500">CLIENT VIEW:</span>
                    <div className="text-white mt-1">[{heData.client_view.map(n => n.toFixed(2)).join(', ')}]</div>
                  </div>
                  <div>
                    <span className="text-slate-500">SERVER VIEW (ENCRYPTED):</span>
                    <div className="text-cyber-green mt-1 truncate">{heData.server_view_snippet}</div>
                  </div>
                </div>
              </motion.div>
            )}
          </div>

          {/* Action Bar */}
          <div className="glass-panel">
            <button onClick={simulateHE}
              className="w-full py-4 bg-cyber-cyan/10 border border-cyber-cyan text-cyber-cyan font-bold rounded hover:bg-cyber-cyan hover:text-slate-900 transition-all tracking-widest">
              RUN FEDERATED TRAINING ROUND
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
