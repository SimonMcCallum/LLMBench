import React, { useState, useMemo } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ReferenceLine } from 'recharts';
import { Info, AlertTriangle, TrendingUp, ShieldAlert, Zap } from 'lucide-react';

const App = () => {
  // State for sliders
  const [p, setP] = useState(0.8);
  const [setterError, setSetterError] = useState(0.02); // Question setter accuracy
  const [humilityError, setHumilityError] = useState(0.03); // Historical error rate
  const [modelNoise, setModelNoise] = useState(0.01); // Base statistical noise

  // Derived Values
  const totalUncertainty = useMemo(() => {
    // Total uncertainty is the sum of all error sources
    return Math.min(0.2, setterError + humilityError + modelNoise);
  }, [setterError, humilityError, modelNoise]);

  // The "Effective Risk" denominator: (1 - p + totalUncertainty)
  // We cap p at 0.999 to avoid division by zero in the chart logic
  const calculateC = (prob, uncertainty) => {
    const risk = Math.max(0.001, (1 - prob) + uncertainty);
    return prob / (4 * risk);
  };

  const calculateReturn = (prob, betC, uncertainty) => {
    // The "True" expected return based on the optimization formula
    // E[S] = p(1+c) - 2(1-p+uncertainty)c^2
    const risk = (1 - prob) + uncertainty;
    return prob * (1 + betC) - 2 * risk * (Math.pow(betC, 2));
  };

  const currentCVal = calculateC(p, totalUncertainty);
  const currentC = currentCVal.toFixed(2);
  const baseC = calculateC(p, 0).toFixed(2); // The "naive" bet without uncertainty
  const expectedReturn = calculateReturn(p, currentCVal, totalUncertainty).toFixed(2);

  // Generate Data for the Charts
  const chartData = useMemo(() => {
    const data = [];
    for (let i = 0.5; i <= 0.99; i += 0.01) {
      const naive = calculateC(i, 0);
      const robust = calculateC(i, totalUncertainty);
      data.push({
        prob: i.toFixed(2),
        naiveC: parseFloat(naive.toFixed(2)),
        robustC: parseFloat(robust.toFixed(2)),
        uncertaintyBand: (totalUncertainty * 100).toFixed(1),
      });
    }
    return data;
  }, [totalUncertainty]);

  return (
    <div className="flex flex-col min-h-screen bg-slate-50 p-4 md:p-8 font-sans text-slate-900">
      <div className="max-w-6xl mx-auto w-full space-y-6">
        
        {/* Header */}
        <div className="bg-white rounded-2xl p-6 shadow-sm border border-slate-200">
          <h1 className="text-2xl font-bold text-slate-800 flex items-center gap-2">
            <Zap className="text-amber-500" /> Confidence & Uncertainty Lab
          </h1>
          <p className="text-slate-500 mt-1">
            Explore how hidden errors "buffer" your betting strategy. As uncertainty rises, your optimal bet $c^*$ becomes more conservative.
          </p>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          
          {/* Controls Panel */}
          <div className="lg:col-span-1 space-y-4">
            <div className="bg-white rounded-2xl p-6 shadow-sm border border-slate-200 space-y-6">
              <h2 className="font-semibold flex items-center gap-2 border-b pb-2">
                <Info size={18} className="text-blue-500" /> Parameters
              </h2>
              
              {/* Base Confidence Slider */}
              <div className="space-y-3">
                <div className="flex justify-between items-center">
                  <label className="text-sm font-medium text-slate-700">Your Confidence ($p$)</label>
                  <span className="bg-blue-100 text-blue-700 px-2 py-1 rounded text-xs font-bold">{(p * 100).toFixed(0)}%</span>
                </div>
                <input 
                  type="range" min="0.5" max="0.99" step="0.01" value={p} 
                  onChange={(e) => setP(parseFloat(e.target.value))}
                  className="w-full h-2 bg-slate-200 rounded-lg appearance-none cursor-pointer accent-blue-600"
                />
              </div>

              {/* Question Setter Accuracy */}
              <div className="space-y-3">
                <div className="flex justify-between items-center">
                  <label className="text-sm font-medium text-slate-700">Question Ambiguity</label>
                  <span className="bg-emerald-100 text-emerald-700 px-2 py-1 rounded text-xs font-bold">{(setterError * 100).toFixed(1)}%</span>
                </div>
                <input 
                  type="range" min="0" max="0.1" step="0.005" value={setterError} 
                  onChange={(e) => setSetterError(parseFloat(e.target.value))}
                  className="w-full h-2 bg-slate-200 rounded-lg appearance-none cursor-pointer accent-emerald-500"
                />
                <p className="text-[10px] text-slate-400 italic leading-tight">Risk that the question is flawed or misleading.</p>
              </div>

              {/* Epistemic Humility */}
              <div className="space-y-3">
                <div className="flex justify-between items-center">
                  <label className="text-sm font-medium text-slate-700">Epistemic Humility</label>
                  <span className="bg-rose-100 text-rose-700 px-2 py-1 rounded text-xs font-bold">{(humilityError * 100).toFixed(1)}%</span>
                </div>
                <input 
                  type="range" min="0" max="0.1" step="0.005" value={humilityError} 
                  onChange={(e) => setHumilityError(parseFloat(e.target.value))}
                  className="w-full h-2 bg-slate-200 rounded-lg appearance-none cursor-pointer accent-rose-500"
                />
                <p className="text-[10px] text-slate-400 italic leading-tight">Frequency of being wrong when "certain" in the past.</p>
              </div>

              {/* Model Noise */}
              <div className="space-y-3">
                <div className="flex justify-between items-center">
                  <label className="text-sm font-medium text-slate-700">Base Model Noise</label>
                  <span className="bg-slate-200 text-slate-700 px-2 py-1 rounded text-xs font-bold">{(modelNoise * 100).toFixed(1)}%</span>
                </div>
                <input 
                  type="range" min="0" max="0.05" step="0.005" value={modelNoise} 
                  onChange={(e) => setModelNoise(parseFloat(e.target.value))}
                  className="w-full h-2 bg-slate-200 rounded-lg appearance-none cursor-pointer accent-slate-600"
                />
              </div>

              <div className="pt-4 border-t border-slate-100">
                <div className="flex justify-between items-center">
                  <span className="text-sm font-bold text-slate-800">Total Uncertainty Buffer:</span>
                  <span className="text-lg font-black text-rose-600">{(totalUncertainty * 100).toFixed(1)}%</span>
                </div>
              </div>
            </div>

            {/* Live Stats */}
            <div className="bg-slate-900 rounded-2xl p-6 shadow-lg text-white space-y-4">
              <h3 className="text-xs uppercase tracking-widest text-slate-400 font-bold">Calculated Strategy</h3>
              
              <div className="grid grid-cols-2 gap-4">
                <div className="p-3 bg-slate-800 rounded-xl border border-slate-700">
                  <p className="text-[10px] text-slate-400">Optimal Bet ($c^*$)</p>
                  <p className="text-2xl font-bold text-amber-400">{currentC}</p>
                </div>
                <div className="p-3 bg-slate-800 rounded-xl border border-slate-700">
                  <p className="text-[10px] text-slate-400">Expected Profit</p>
                  <p className="text-2xl font-bold text-emerald-400">{expectedReturn}</p>
                </div>
              </div>

              {parseFloat(baseC) > parseFloat(currentC) * 1.5 && (
                <div className="flex items-start gap-2 text-xs bg-rose-500/10 text-rose-300 p-2 rounded-lg border border-rose-500/20">
                  <AlertTriangle size={14} className="shrink-0 mt-0.5" />
                  <span>Uncertainty is reducing your bet by {Math.round((1 - parseFloat(currentC)/parseFloat(baseC)) * 100)}% vs the naive model.</span>
                </div>
              )}
            </div>
          </div>

          {/* Visualization Panel */}
          <div className="lg:col-span-2 space-y-6">
            
            {/* Betting Strategy Chart */}
            <div className="bg-white rounded-2xl p-6 shadow-sm border border-slate-200">
              <div className="flex justify-between items-center mb-6">
                <div>
                  <h3 className="font-bold text-slate-800 flex items-center gap-2">
                    <TrendingUp size={18} className="text-blue-500" /> Optimal Bet Curve ($c^*$)
                  </h3>
                  <p className="text-xs text-slate-500">Comparing Naive model vs your Robust model with uncertainty.</p>
                </div>
                <div className="flex gap-4 text-[10px] font-bold">
                  <div className="flex items-center gap-1"><div className="w-2 h-2 rounded-full bg-slate-300"></div> NAIVE</div>
                  <div className="flex items-center gap-1"><div className="w-2 h-2 rounded-full bg-blue-600"></div> ROBUST</div>
                </div>
              </div>
              
              <div className="h-[300px] w-full">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={chartData} margin={{ top: 5, right: 30, left: 0, bottom: 5 }}>
                    <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#f1f5f9" />
                    <XAxis 
                      dataKey="prob" 
                      label={{ value: 'Confidence (p)', position: 'insideBottom', offset: -5 }} 
                      tick={{fontSize: 10}}
                      stroke="#94a3b8"
                    />
                    <YAxis 
                      label={{ value: 'Bet Size (c)', angle: -90, position: 'insideLeft', offset: 15 }} 
                      tick={{fontSize: 10}}
                      stroke="#94a3b8"
                    />
                    <Tooltip 
                      contentStyle={{ borderRadius: '12px', border: 'none', boxShadow: '0 10px 15px -3px rgba(0,0,0,0.1)' }}
                    />
                    <ReferenceLine x={p.toFixed(2)} stroke="#cbd5e1" strokeDasharray="5 5" label={{ value: 'You', position: 'top', fill: '#64748b', fontSize: 10 }} />
                    
                    <Line 
                      type="monotone" 
                      dataKey="naiveC" 
                      stroke="#cbd5e1" 
                      strokeWidth={2} 
                      strokeDasharray="4 4"
                      dot={false}
                      name="Naive (No Error)"
                    />
                    <Line 
                      type="monotone" 
                      dataKey="robustC" 
                      stroke="#2563eb" 
                      strokeWidth={4} 
                      dot={false}
                      name="Robust (Your Strategy)"
                    />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </div>

            {/* Uncertainty Breakdown Area */}
            <div className="bg-white rounded-2xl p-6 shadow-sm border border-slate-200">
               <h3 className="font-bold text-slate-800 mb-4 flex items-center gap-2">
                <ShieldAlert size={18} className="text-rose-500" /> The "Invisible" Risk Floor
              </h3>
              <p className="text-sm text-slate-600 mb-6">
                This visualizes the risk "buffer" (1 - p + Uncertainty) that prevents you from betting infinitely on "sure things."
              </p>
              
              <div className="h-[180px] w-full bg-slate-50 rounded-xl overflow-hidden relative">
                <div className="absolute inset-0 flex">
                  {/* Confidence Zone */}
                  <div 
                    className="h-full bg-blue-500 transition-all duration-300 flex items-center justify-center text-white text-[10px] font-bold overflow-hidden" 
                    style={{ width: `${p * 100}%` }}
                  >
                    CONFIDENCE: {(p*100).toFixed(0)}%
                  </div>
                  
                  {/* Statistical Risk Zone */}
                  <div 
                    className="h-full bg-slate-300 transition-all duration-300 flex items-center justify-center text-slate-600 text-[10px] font-bold overflow-hidden" 
                    style={{ width: `${(1-p) * 100}%` }}
                  >
                    EVENT RISK
                  </div>

                  {/* Uncertainty Buffer Zone */}
                  <div 
                    className="h-full bg-rose-400 transition-all duration-300 flex items-center justify-center text-white text-[10px] font-bold overflow-hidden relative group" 
                    style={{ width: `${totalUncertainty * 100}%` }}
                  >
                    <span className="z-10">ERROR BUFFER</span>
                    <div className="absolute inset-0 bg-rose-500 animate-pulse opacity-20"></div>
                  </div>
                </div>
              </div>

              <div className="mt-4 grid grid-cols-1 md:grid-cols-2 gap-4 text-xs text-slate-500 italic">
                <div className="flex gap-2">
                  <span className="font-bold text-blue-600 not-italic">Strategy:</span>
                  "The more ambiguous the setter or the worse your history, the wider the red buffer gets, capping your bet."
                </div>
                <div className="flex gap-2">
                  <span className="font-bold text-rose-600 not-italic">Observation:</span>
                  "Even if Confidence = 100%, the Error Buffer stays, keeping the denominator {'>'} 0."
                </div>
              </div>
            </div>

          </div>
        </div>
      </div>
    </div>
  );
};

export default App;