'use client';

import React, { useState, useEffect, useRef } from 'react';
import {
  MessageSquare,
  X,
  Send,
  Flame,
  Shield,
  Home,
  Sparkles,
  Bot,
  User,
  Wrench,
  ChevronDown,
  RotateCcw,
  Minimize2,
  ExternalLink,
} from 'lucide-react';

interface AICopilotDrawerProps {
  selectedRegion: string;
}

interface Message {
  id: string;
  role: 'user' | 'assistant' | 'system';
  content: string;
  reasoning?: string;
  executedTools?: Array<{ tool: string; args: any; result_summary: string }>;
  timestamp: string;
}

export default function AICopilotDrawer({ selectedRegion }: AICopilotDrawerProps) {
  const [isOpen, setIsOpen] = useState<boolean>(false);
  const [persona, setPersona] = useState<'resident' | 'firefighter'>('resident');
  const [inputMessage, setInputMessage] = useState<string>('');
  const [messages, setMessages] = useState<Message[]>([]);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [prompts, setPrompts] = useState<string[]>([]);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  // Default greeting when drawer opens or persona changes
  useEffect(() => {
    fetch(`http://localhost:8000/api/ai/suggested-prompts?persona=${persona}`)
      .then((res) => res.json())
      .then((data) => {
        if (data.success && data.prompts) {
          setPrompts(data.prompts);
        }
      })
      .catch(() => {
        setPrompts(
          persona === 'firefighter'
            ? [
                'What is the 130ft radiant heat contagion risk for Crestmoor Canyon?',
                'Fetch live NOAA wind vectors and CDEC fuel moisture.',
                'Compare Bellwether 100m dynamic risk against CAL FIRE Moderate FHSZ.',
              ]
            : [
                'How can I qualify for mandatory California insurance discounts?',
                'What is the difference between Zone 0 (0-5ft) and Zone 1 defensible space?',
                'Why did my insurance company non-renew my policy in the WUI?',
              ]
        );
      });

    const greetingMessage: Message = {
      id: 'welcome-' + persona,
      role: 'assistant',
      content:
        persona === 'firefighter'
          ? `### 🚒 Incident Commander & Tactical Fire Operations Active
I am your **Wildfire AI Copilot** connected directly to the GIS platform. I have live access to:
* **NOAA HRRR** live hourly wind vectors & gusts
* **CDEC / RAWS** dead fuel moisture (DFM) & live fuel moisture (LFMC)
* **130ft Radiant Heat Contagion** calculation (Saylors S-Ratio saved property valuation)
* **Bellwether 100m** AI hazard probability vs. **CAL FIRE FHSZ**

What incident coordinates or tactical conditions would you like to evaluate?`
          : `### 🏡 Resident & Community Wildfire Safety Active
Hello! I am your **Wildfire AI Assistant**. I can help you:
* **Check your property risk** against predictive AI hazard models
* **Lower your insurance premiums** using California's *Safer from Wildfires* (10 CCR § 2644.9) discounts
* **Audit your home hardening**: Class A roofing, 1/16" ember-resistant vents, and Zone 0 (0–5ft) defensible space
* **Appeal unfair policy non-renewals** with verified mitigation evidence

Ask me any question below or click a suggested topic to begin!`,
      timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
    };

    setMessages([greetingMessage]);
  }, [persona]);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const handleSendMessage = async (userText: string) => {
    if (!userText.trim() || isLoading) return;

    const userMessage: Message = {
      id: Date.now().toString(),
      role: 'user',
      content: userText.trim(),
      timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
    };

    setMessages((prev) => [...prev, userMessage]);
    setInputMessage('');
    setIsLoading(true);

    try {
      const history = messages
        .filter((m) => m.role !== 'system')
        .slice(-6)
        .map((m) => ({ role: m.role, content: m.content }));

      history.push({ role: 'user', content: userText.trim() });

      const response = await fetch('http://localhost:8000/api/ai/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          messages: history,
          persona: persona,
          context: { region: selectedRegion },
        }),
      });

      const resData = await response.json();

      if (resData.success) {
        const assistantMsg: Message = {
          id: (Date.now() + 1).toString(),
          role: 'assistant',
          content: resData.content,
          reasoning: resData.reasoning,
          executedTools: resData.executed_tools,
          timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
        };
        setMessages((prev) => [...prev, assistantMsg]);
      } else {
        const errorMsg: Message = {
          id: (Date.now() + 1).toString(),
          role: 'assistant',
          content: `⚠️ **AI Response Error**: ${resData.error || 'Failed to generate answer. Please verify backend connection.'}`,
          timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
        };
        setMessages((prev) => [...prev, errorMsg]);
      }
    } catch (err: any) {
      setMessages((prev) => [
        ...prev,
        {
          id: (Date.now() + 1).toString(),
          role: 'assistant',
          content: `⚠️ **Connection Error**: Could not connect to the backend AI agent service at \`http://localhost:8000/api/ai/chat\`.`,
          timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
        },
      ]);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="fixed bottom-5 right-5 z-40 flex flex-col items-end">
      {/* Floating Action Button (Collapsed state) */}
      {!isOpen && (
        <button
          onClick={() => setIsOpen(true)}
          className="group flex items-center gap-3 px-5 py-3.5 rounded-full bg-gradient-to-r from-orange-500 via-amber-600 to-orange-600 hover:from-orange-400 hover:to-orange-500 text-slate-950 font-extrabold shadow-2xl shadow-orange-500/30 border border-orange-400/50 transition-all duration-300 hover:scale-105 active:scale-95 glow-orange"
        >
          <div className="relative">
            <Bot className="w-6 h-6 text-slate-950 animate-bounce" />
            <span className="absolute -top-1 -right-1 w-2.5 h-2.5 rounded-full bg-emerald-400 border-2 border-slate-950 animate-pulse" />
          </div>
          <div className="text-left">
            <div className="text-xs uppercase tracking-wider text-slate-950 font-black">AI Wildfire Copilot</div>
            <div className="text-[10px] text-slate-900 font-medium">MiniMax-M3 Reasoning & Live GIS</div>
          </div>
        </button>
      )}

      {/* Expanded Chat Drawer */}
      {isOpen && (
        <div className="w-[430px] max-w-[95vw] h-[610px] max-h-[88vh] glass-panel bg-slate-900/95 backdrop-blur-2xl border border-slate-700/80 rounded-3xl shadow-2xl flex flex-col overflow-hidden animate-in slide-in-from-bottom-5 duration-200">
          
          {/* Header */}
          <div className="p-4 border-b border-slate-800 bg-slate-950/60 shrink-0 space-y-2.5">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2.5">
                <div className="p-2 rounded-xl bg-orange-500/20 text-orange-400 border border-orange-500/30">
                  <Sparkles className="w-4 h-4" />
                </div>
                <div>
                  <h3 className="text-sm font-bold text-slate-100 flex items-center gap-2">
                    Wildfire AI Copilot
                    <span className="px-2 py-0.5 rounded-full text-[9px] font-bold bg-emerald-500/20 text-emerald-400 border border-emerald-500/30">
                      MiniMax-M3
                    </span>
                  </h3>
                  <p className="text-[10px] text-slate-400">Autonomous GIS Tool-Calling & Reasoning</p>
                </div>
              </div>

              <div className="flex items-center gap-1">
                <button
                  onClick={() => setIsOpen(false)}
                  className="p-1.5 rounded-lg text-slate-400 hover:text-slate-100 hover:bg-slate-800 transition-colors"
                  title="Minimize Copilot"
                >
                  <Minimize2 className="w-4 h-4" />
                </button>
              </div>
            </div>

            {/* Persona Switcher Buttons */}
            <div className="grid grid-cols-2 gap-1.5 bg-slate-950 p-1 rounded-xl border border-slate-800">
              <button
                onClick={() => setPersona('resident')}
                className={`py-1.5 px-2.5 rounded-lg text-xs font-bold transition-all flex items-center justify-center gap-1.5 ${
                  persona === 'resident'
                    ? 'bg-emerald-500/20 text-emerald-300 border border-emerald-500/40 shadow-sm'
                    : 'text-slate-400 hover:text-slate-200'
                }`}
              >
                <Home className="w-3.5 h-3.5" />
                🏡 Resident Mode
              </button>

              <button
                onClick={() => setPersona('firefighter')}
                className={`py-1.5 px-2.5 rounded-lg text-xs font-bold transition-all flex items-center justify-center gap-1.5 ${
                  persona === 'firefighter'
                    ? 'bg-orange-500/20 text-orange-300 border border-orange-500/40 shadow-sm'
                    : 'text-slate-400 hover:text-slate-200'
                }`}
              >
                <Shield className="w-3.5 h-3.5" />
                🚒 Firefighter Mode
              </button>
            </div>
          </div>

          {/* Messages Scroll Area */}
          <div className="flex-1 overflow-y-auto p-4 space-y-3.5 text-xs">
            {messages.map((m) => (
              <div
                key={m.id}
                className={`flex gap-2.5 ${m.role === 'user' ? 'justify-end' : 'justify-start'}`}
              >
                {m.role === 'assistant' && (
                  <div className="w-7 h-7 rounded-xl bg-orange-500/20 border border-orange-500/30 text-orange-400 flex items-center justify-center shrink-0 mt-0.5">
                    <Bot className="w-4 h-4" />
                  </div>
                )}

                <div
                  className={`max-w-[85%] rounded-2xl p-3.5 space-y-2 leading-relaxed ${
                    m.role === 'user'
                      ? 'bg-orange-500/20 border border-orange-500/40 text-slate-100 rounded-tr-none'
                      : 'bg-slate-950/70 border border-slate-800 text-slate-200 rounded-tl-none'
                  }`}
                >
                  {/* Tool Executions Badge */}
                  {m.executedTools && m.executedTools.length > 0 && (
                    <div className="flex flex-wrap items-center gap-1 pb-1.5 border-b border-slate-800/80 text-[10px] text-orange-400 font-mono">
                      <Wrench className="w-3 h-3 text-orange-400" />
                      <span>GIS Tools Executed:</span>
                      {m.executedTools.map((t, i) => (
                        <span key={i} className="px-1.5 py-0.5 rounded bg-orange-500/10 border border-orange-500/20 font-semibold">
                          {t.tool}
                        </span>
                      ))}
                    </div>
                  )}

                  {/* Message Content */}
                  <div className="prose prose-invert prose-xs max-w-none space-y-1.5 text-xs text-slate-200">
                    {m.content.split('\n').map((line, idx) => {
                      if (line.startsWith('### ')) {
                        return <h4 key={idx} className="font-bold text-sm text-orange-400 mt-2 mb-1">{line.replace('### ', '')}</h4>;
                      }
                      if (line.startsWith('## ')) {
                        return <h3 key={idx} className="font-extrabold text-sm text-slate-100 mt-2.5 mb-1">{line.replace('## ', '')}</h3>;
                      }
                      if (line.startsWith('* ') || line.startsWith('- ')) {
                        return (
                          <div key={idx} className="flex items-start gap-1.5 ml-2 text-[11px] text-slate-300">
                            <span className="text-orange-400 mt-1">•</span>
                            <span>{line.replace(/^(\*|-)\s+/, '')}</span>
                          </div>
                        );
                      }
                      return <p key={idx} className="text-[11px] leading-relaxed text-slate-300">{line}</p>;
                    })}
                  </div>

                  <div className="text-[9px] text-slate-500 text-right">{m.timestamp}</div>
                </div>

                {m.role === 'user' && (
                  <div className="w-7 h-7 rounded-xl bg-slate-800 border border-slate-700 text-slate-300 flex items-center justify-center shrink-0 mt-0.5">
                    <User className="w-4 h-4" />
                  </div>
                )}
              </div>
            ))}

            {isLoading && (
              <div className="flex items-center gap-2 text-orange-400 text-xs p-3 rounded-2xl bg-slate-950/60 border border-slate-800 animate-pulse">
                <Flame className="w-4 h-4 animate-spin" />
                <span>MiniMax-M3 is querying GIS datasets and analyzing telemetry...</span>
              </div>
            )}
            <div ref={messagesEndRef} />
          </div>

          {/* Quick Action Prompt Pills */}
          <div className="px-3.5 py-2 border-t border-slate-800/80 bg-slate-950/40 shrink-0">
            <div className="text-[10px] uppercase font-bold text-slate-400 mb-1 flex items-center gap-1">
              <Sparkles className="w-3 h-3 text-amber-400" /> Suggested Quick Prompts:
            </div>
            <div className="flex gap-1.5 overflow-x-auto pb-1 no-scrollbar">
              {prompts.slice(0, 3).map((p, idx) => (
                <button
                  key={idx}
                  onClick={() => handleSendMessage(p)}
                  className="px-2.5 py-1 rounded-full bg-slate-800/90 hover:bg-slate-800 hover:border-orange-500/50 border border-slate-700 text-[10px] text-slate-300 font-medium whitespace-nowrap transition-colors"
                >
                  {p}
                </button>
              ))}
            </div>
          </div>

          {/* Footer Input Area */}
          <div className="p-3 border-t border-slate-800 bg-slate-950/80 shrink-0">
            <form
              onSubmit={(e) => {
                e.preventDefault();
                handleSendMessage(inputMessage);
              }}
              className="flex items-center gap-2"
            >
              <input
                type="text"
                value={inputMessage}
                onChange={(e) => setInputMessage(e.target.value)}
                placeholder={
                  persona === 'firefighter'
                    ? "Ask tactical fire conditions (e.g. 'Live wind & DFM')..."
                    : "Ask homeowner safety (e.g. 'How to get insurance discount')..."
                }
                className="flex-1 bg-slate-900 border border-slate-700 focus:border-orange-500 rounded-xl px-3.5 py-2.5 text-xs text-slate-100 placeholder-slate-500 outline-none transition-all"
                disabled={isLoading}
              />
              <button
                type="submit"
                disabled={isLoading || !inputMessage.trim()}
                className="p-2.5 rounded-xl bg-orange-500 hover:bg-orange-400 disabled:opacity-40 disabled:cursor-not-allowed text-slate-950 font-bold transition-all shrink-0"
              >
                <Send className="w-4 h-4" />
              </button>
            </form>
          </div>
        </div>
      )}
    </div>
  );
}
