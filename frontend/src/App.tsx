import React, { useState, useEffect } from 'react';
import { 
  Activity, AlertTriangle, CheckCircle, Info, RefreshCw, BarChart2, 
  Database, Download, Sun, Moon, Layers, LineChart, ShieldAlert, Cpu, Gauge, Radio, Code, X, FileText
} from 'lucide-react';
import type { MachineDataInput, PredictionResponse } from './services/api';
import { predictFailure } from './services/api';

const RANGES = {
  "Air temperature [K]": { min: 295.3, max: 304.5 },
  "Process temperature [K]": { min: 305.7, max: 313.8 },
  "Rotational speed [rpm]": { min: 1168, max: 2886 },
  "Torque [Nm]": { min: 3.8, max: 76.6 },
  "Tool wear [min]": { min: 0, max: 253 },
};

const DEFAULT_STATE: MachineDataInput = {
  "Type": 'M',
  "Air temperature [K]": 298.1,
  "Process temperature [K]": 308.6,
  "Rotational speed [rpm]": 1551,
  "Torque [Nm]": 42.8,
  "Tool wear [min]": 0,
};

type GraphTab = 'shap' | 'confusion' | 'roc' | 'correlation' | 'history';

const App: React.FC = () => {
  const [formData, setFormData] = useState<MachineDataInput>(DEFAULT_STATE);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<PredictionResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loadedSampleInfo, setLoadedSampleInfo] = useState<string | null>(null);
  const [darkMode, setDarkMode] = useState<boolean>(true);
  const [activeTab, setActiveTab] = useState<GraphTab>('shap');
  const [currentTime, setCurrentTime] = useState<string>('');
  const [showTechModal, setShowTechModal] = useState<boolean>(false);

  useEffect(() => {
    const updateTime = () => {
      const now = new Date();
      setCurrentTime(now.toLocaleTimeString('en-US', { hour12: false, hour: '2-digit', minute: '2-digit', second: '2-digit' }));
    };
    updateTime();
    const interval = setInterval(updateTime, 1000);
    return () => clearInterval(interval);
  }, []);

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>) => {
    const { name, value } = e.target;
    setFormData((prev) => ({
      ...prev,
      [name]: name === 'Type' ? value : (value === '' ? ('' as any) : parseFloat(value)),
    }));
    setLoadedSampleInfo(null);
  };

  const handleReset = () => {
    setFormData(DEFAULT_STATE);
    setResult(null);
    setError(null);
    setLoadedSampleInfo(null);
  };

  const loadNormalSample = () => {
    setFormData({
      "Type": 'M',
      "Air temperature [K]": 298.1,
      "Process temperature [K]": 308.6,
      "Rotational speed [rpm]": 1551,
      "Torque [Nm]": 42.8,
      "Tool wear [min]": 0,
    });
    setResult(null);
    setError(null);
    setLoadedSampleInfo("Normal Telemetry Sample Loaded");
  };

  const loadFailureSample = () => {
    setFormData({
      "Type": 'L',
      "Air temperature [K]": 298.9,
      "Process temperature [K]": 309.1,
      "Rotational speed [rpm]": 2861,
      "Torque [Nm]": 4.6,
      "Tool wear [min]": 143,
    });
    setResult(null);
    setError(null);
    setLoadedSampleInfo("Power Failure Sample (PWF) Loaded");
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setResult(null);

    for (const [key, value] of Object.entries(formData)) {
      if (value === '' || Number.isNaN(value)) {
        setError(`Please provide a valid numeric value for ${key}`);
        setLoading(false);
        return;
      }
    }

    try {
      const res = await predictFailure(formData);
      setResult(res);
    } catch (err: any) {
      setError(err.message || 'An unexpected network error occurred.');
    } finally {
      setLoading(false);
    }
  };

  const renderInput = (label: string, name: keyof typeof RANGES, step: string) => {
    const val = formData[name];
    const range = RANGES[name];
    const isOutOfRange = 
      typeof val === 'number' && 
      (val < range.min || val > range.max);

    const pct = typeof val === 'number'
      ? Math.max(0, Math.min(100, ((val - range.min) / (range.max - range.min)) * 100))
      : 0;

    return (
      <div className="space-y-1.5 relative">
        <div className="flex justify-between items-center">
          <label className={`text-xs font-bold uppercase tracking-wider ${darkMode ? 'text-slate-300' : 'text-slate-700'}`}>
            {label}
          </label>
          <span className={`text-[10px] font-mono font-semibold ${darkMode ? 'text-sky-400' : 'text-sky-700'}`}>
            [{range.min} - {range.max}]
          </span>
        </div>

        <div className="relative">
          <input
            type="number"
            step={step}
            name={name}
            value={val}
            onChange={handleInputChange}
            className={`w-full border rounded-lg px-4 py-2.5 font-mono text-sm focus:outline-none transition-all shadow-sm ${
              darkMode 
                ? `bg-[#0F172A] text-slate-100 ${isOutOfRange ? 'border-amber-500/80 focus:border-amber-500 focus:ring-1 focus:ring-amber-500' : 'border-slate-800 focus:border-sky-400 focus:ring-1 focus:ring-sky-400'}` 
                : `bg-[#F8FAFC] text-slate-900 ${isOutOfRange ? 'border-amber-500 focus:border-amber-500 focus:ring-1 focus:ring-amber-500' : 'border-slate-300 focus:border-sky-600 focus:ring-1 focus:ring-sky-600'}`
            }`}
            required
          />
        </div>

        {/* Mini Range Percentage Bar */}
        <div className="space-y-1 pt-0.5">
          <div className={`w-full h-1.5 rounded-full overflow-hidden ${darkMode ? 'bg-slate-800' : 'bg-slate-200'}`}>
            <div 
              className={`h-full transition-all duration-300 ${
                isOutOfRange 
                  ? 'bg-amber-500' 
                  : pct > 80 ? 'bg-amber-500' : darkMode ? 'bg-sky-400' : 'bg-sky-600'
              }`}
              style={{ width: `${pct}%` }}
            />
          </div>
          {isOutOfRange && (
            <div className="flex items-center text-[10px] text-amber-600 dark:text-amber-400 font-semibold">
              <AlertTriangle className="w-3 h-3 mr-1 shrink-0" />
              Outside model training distribution
            </div>
          )}
        </div>
      </div>
    );
  };

  const graphTabInfo: Record<GraphTab, { title: string; desc: string; src: string }> = {
    shap: {
      title: 'Global SHAP Feature Importance',
      desc: 'Explains feature impact across test predictions. Tool Wear, Torque, and Rotational Speed dominate failure indicators.',
      src: '/shap_summary.png',
    },
    confusion: {
      title: 'Confusion Matrix (Evaluation Benchmark)',
      desc: 'Evaluates True Positives, True Negatives, False Positives, and False Negatives of the ANN classifier.',
      src: '/confusion_matrix_final.png',
    },
    roc: {
      title: 'ROC Curve (AUC = 97.16%)',
      desc: 'Diagnostic performance curve of the ANN model across decision thresholds.',
      src: '/roc_curve.png',
    },
    correlation: {
      title: 'Feature Correlation Matrix',
      desc: 'Linear relationships between input variables and physical machine failures.',
      src: '/correlation_heatmap.png',
    },
    history: {
      title: 'ANN Training History',
      desc: 'Loss and accuracy trajectory over training epochs.',
      src: '/training_history_accuracy.png',
    }
  };

  return (
    <div className={`min-h-screen transition-colors duration-300 font-sans p-3 sm:p-6 md:p-8 ${
      darkMode 
        ? 'bg-[#0B1120] text-slate-200' 
        : 'bg-[#F1F5F9] text-slate-800'
    }`}>
      <div className="max-w-7xl mx-auto space-y-6">
        
        {/* Top Telemetry Control Header */}
        <div className={`flex flex-wrap items-center justify-between px-4 py-2.5 rounded-xl border text-xs font-mono gap-3 shadow-md ${
          darkMode ? 'bg-[#151F32] border-slate-800 text-slate-300' : 'bg-[#0F172A] border-slate-900 text-slate-200'
        }`}>
          <div className="flex items-center space-x-3">
            <span className="flex items-center text-emerald-400 font-bold">
              <Radio className="w-3.5 h-3.5 mr-1.5 animate-pulse text-emerald-400" />
              TELEMETRY ONLINE
            </span>
            <span className="hidden sm:inline text-slate-600">|</span>
            <span className="hidden sm:inline flex items-center text-sky-400">
              <Cpu className="w-3.5 h-3.5 mr-1" /> Keras ANN Core v1.0
            </span>
          </div>

          <div className="flex items-center space-x-4">
            <span className="flex items-center text-amber-300">
              <Gauge className="w-3.5 h-3.5 mr-1 text-amber-400" /> Dataset: AI4I 2020
            </span>
            <span className="text-slate-600">|</span>
            <span className="font-bold text-sky-300">{currentTime || '00:00:00'} UTC</span>
          </div>
        </div>

        {/* Main Title & Theme Controls */}
        <header className={`flex flex-col sm:flex-row sm:items-center justify-between pb-5 border-b gap-4 ${
          darkMode ? 'border-slate-800' : 'border-slate-300'
        }`}>
          <div className="flex items-center space-x-4">
            <div className={`p-3 rounded-xl border ${
              darkMode 
                ? 'bg-[#151F32] border-slate-700 text-sky-400' 
                : 'bg-white border-slate-300 text-sky-600 shadow-md'
            }`}>
              <Activity className="w-8 h-8" />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <h1 className={`text-2xl sm:text-3xl font-black tracking-tight ${
                  darkMode ? 'text-slate-100' : 'text-slate-900'
                }`}>
                  Predictive Pulse
                </h1>
                <span className={`text-[10px] font-bold px-2 py-0.5 rounded-full uppercase tracking-wide border ${
                  darkMode 
                    ? 'bg-sky-500/10 text-sky-400 border-sky-500/20' 
                    : 'bg-sky-100 text-sky-800 border-sky-300'
                }`}>
                  Industrial Edition
                </span>
              </div>
              <p className={`text-xs sm:text-sm mt-0.5 font-medium ${darkMode ? 'text-slate-400' : 'text-slate-600'}`}>
                Explainable Deep Learning Framework for Industrial Predictive Maintenance
              </p>
            </div>
          </div>
          
          <div className="flex items-center gap-2.5 shrink-0">
            {/* Tech Specs Modal Button */}
            <button
              onClick={() => setShowTechModal(true)}
              className={`px-3 py-2 rounded-lg border transition-all flex items-center gap-1.5 text-xs font-bold shadow-sm ${
                darkMode 
                  ? 'bg-[#151F32] hover:bg-slate-800 border-slate-700 text-sky-400' 
                  : 'bg-white hover:bg-slate-100 border-slate-300 text-sky-700 shadow'
              }`}
            >
              <Code className="w-4 h-4 text-sky-500" />
              <span>Technical Specs</span>
            </button>

            {/* Theme Toggle Button */}
            <button
              onClick={() => setDarkMode(!darkMode)}
              className={`px-3 py-2 rounded-lg border transition-all flex items-center gap-2 text-xs font-bold shadow-sm ${
                darkMode 
                  ? 'bg-[#151F32] hover:bg-slate-800 border-slate-700 text-amber-400' 
                  : 'bg-white hover:bg-slate-100 border-slate-300 text-slate-900 shadow'
              }`}
              title="Toggle Dark Command Center / Light Dashboard"
            >
              {darkMode ? (
                <>
                  <Sun className="w-4 h-4 text-amber-400" />
                  <span>Light Dashboard</span>
                </>
              ) : (
                <>
                  <Moon className="w-4 h-4 text-sky-600" />
                  <span>Industrial Dark</span>
                </>
              )}
            </button>

            <button 
              onClick={loadNormalSample} 
              className={`text-xs px-3 py-2 rounded-lg border flex items-center transition-all font-semibold ${
                darkMode 
                  ? 'bg-[#151F32] hover:bg-slate-800 border-slate-700 text-slate-200' 
                  : 'bg-white hover:bg-slate-100 border-slate-300 text-slate-800 shadow-sm'
              }`}
            >
              <Download className="w-3.5 h-3.5 mr-1 text-emerald-500" /> Normal Sample
            </button>
            <button 
              onClick={loadFailureSample} 
              className={`text-xs px-3 py-2 rounded-lg border flex items-center transition-all font-semibold ${
                darkMode 
                  ? 'bg-[#151F32] hover:bg-slate-800 border-slate-700 text-slate-200' 
                  : 'bg-white hover:bg-slate-100 border-slate-300 text-slate-800 shadow-sm'
              }`}
            >
              <Download className="w-3.5 h-3.5 mr-1 text-amber-500" /> Failure Sample
            </button>
          </div>
        </header>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          
          {/* Main Input Form Section */}
          <div className="lg:col-span-2 space-y-6">
            
            {/* Input Card Container */}
            <div className={`border rounded-xl p-5 sm:p-6 shadow-xl relative transition-all ${
              darkMode 
                ? 'bg-[#151F32] border-slate-800' 
                : 'bg-white border-slate-300 shadow-lg shadow-slate-300/30'
            }`}>
              
              <div className={`flex justify-between items-center mb-5 pb-3 border-b ${
                darkMode ? 'border-slate-800' : 'border-slate-200'
              }`}>
                <h2 className={`text-base font-extrabold flex items-center uppercase tracking-wider ${
                  darkMode ? 'text-slate-100' : 'text-slate-900'
                }`}>
                  <BarChart2 className="w-5 h-5 mr-2 text-sky-500" />
                  Machine Telemetry Inputs
                </h2>
                {loadedSampleInfo && (
                  <span className={`text-[11px] font-mono font-bold px-2.5 py-1 rounded border animate-pulse ${
                    darkMode 
                      ? 'bg-sky-500/10 text-sky-400 border-sky-500/30' 
                      : 'bg-sky-100 text-sky-900 border-sky-300'
                  }`}>
                    {loadedSampleInfo}
                  </span>
                )}
              </div>
              
              <form onSubmit={handleSubmit} className="space-y-6">
                <div className="grid grid-cols-1 md:grid-cols-2 gap-x-6 gap-y-5">
                  
                  {/* Quality Variant */}
                  <div className="space-y-1.5">
                    <label className={`text-xs font-bold uppercase tracking-wider ${darkMode ? 'text-slate-300' : 'text-slate-700'}`}>
                      Product Quality Type
                    </label>
                    <select
                      name="Type"
                      value={formData['Type']}
                      onChange={handleInputChange}
                      className={`w-full border rounded-lg px-4 py-2.5 font-mono text-sm focus:outline-none transition-all shadow-sm ${
                        darkMode 
                          ? 'bg-[#0F172A] border-slate-800 text-slate-100 focus:border-sky-400 focus:ring-1 focus:ring-sky-400' 
                          : 'bg-[#F8FAFC] border-slate-300 text-slate-900 focus:border-sky-600 focus:ring-1 focus:ring-sky-600'
                      }`}
                    >
                      <option value="L">Low Variant (L)</option>
                      <option value="M">Medium Variant (M)</option>
                      <option value="H">High Variant (H)</option>
                    </select>
                    <div className={`text-[10px] font-medium mt-1 ${darkMode ? 'text-slate-500' : 'text-slate-500'}`}>
                      Categorical product variant classification
                    </div>
                  </div>

                  {renderInput("Air Temperature [K]", "Air temperature [K]", "0.1")}
                  {renderInput("Process Temperature [K]", "Process temperature [K]", "0.1")}
                  {renderInput("Rotational Speed [rpm]", "Rotational speed [rpm]", "1")}
                  {renderInput("Torque [Nm]", "Torque [Nm]", "0.1")}
                  {renderInput("Tool Wear [min]", "Tool wear [min]", "1")}
                  
                </div>

                <div className={`flex space-x-4 pt-4 border-t mt-6 ${darkMode ? 'border-slate-800' : 'border-slate-200'}`}>
                  <button
                    type="submit"
                    disabled={loading}
                    className={`flex-1 font-extrabold py-3.5 px-6 rounded-lg transition-all flex justify-center items-center shadow-lg uppercase tracking-wider text-xs disabled:opacity-50 disabled:cursor-not-allowed ${
                      darkMode 
                        ? 'bg-sky-500 hover:bg-sky-400 text-slate-950 shadow-sky-500/20' 
                        : 'bg-sky-600 hover:bg-sky-500 text-white shadow-sky-600/30'
                    }`}
                  >
                    {loading ? (
                      <>
                        <RefreshCw className="w-4 h-4 mr-2 animate-spin" />
                        Running ANN Inference...
                      </>
                    ) : (
                      'PREDICT FAILURE RISK'
                    )}
                  </button>
                  <button
                    type="button"
                    onClick={handleReset}
                    className={`font-bold py-3.5 px-6 rounded-lg border transition-colors text-xs uppercase tracking-wider ${
                      darkMode 
                        ? 'bg-[#0F172A] hover:bg-slate-800 border-slate-700 text-slate-300' 
                        : 'bg-slate-200 hover:bg-slate-300 border-slate-300 text-slate-800'
                    }`}
                  >
                    Reset
                  </button>
                </div>
              </form>

              {/* Error Box */}
              {error && (
                <div className="mt-6 bg-red-500/10 border border-red-500/30 text-red-600 dark:text-red-400 p-4 rounded-lg flex items-start">
                  <ShieldAlert className="w-5 h-5 mr-3 shrink-0 mt-0.5 text-red-500" />
                  <div>
                    <h4 className="font-bold text-sm">Prediction Request Error</h4>
                    <p className="text-xs mt-1 font-mono">{error}</p>
                  </div>
                </div>
              )}
            </div>

            {/* Explainability & Graph Gallery */}
            <div className={`border rounded-xl p-5 sm:p-6 shadow-xl transition-all ${
              darkMode 
                ? 'bg-[#151F32] border-slate-800' 
                : 'bg-white border-slate-300 shadow-lg shadow-slate-300/30'
            }`}>
              
              {/* Tab Navigation */}
              <div className={`flex flex-wrap gap-2 pb-4 mb-4 border-b ${darkMode ? 'border-slate-800' : 'border-slate-200'}`}>
                <button
                  onClick={() => setActiveTab('shap')}
                  className={`text-xs font-extrabold px-3 py-2 rounded-lg flex items-center gap-1.5 transition-all ${
                    activeTab === 'shap'
                      ? darkMode ? 'bg-sky-500 text-slate-950 shadow-md' : 'bg-sky-600 text-white shadow-md'
                      : darkMode ? 'bg-[#0F172A] text-slate-400 hover:text-slate-200' : 'bg-slate-100 text-slate-700 hover:bg-slate-200'
                  }`}
                >
                  <Layers className="w-3.5 h-3.5" /> SHAP Importance
                </button>
                <button
                  onClick={() => setActiveTab('confusion')}
                  className={`text-xs font-extrabold px-3 py-2 rounded-lg flex items-center gap-1.5 transition-all ${
                    activeTab === 'confusion'
                      ? darkMode ? 'bg-sky-500 text-slate-950 shadow-md' : 'bg-sky-600 text-white shadow-md'
                      : darkMode ? 'bg-[#0F172A] text-slate-400 hover:text-slate-200' : 'bg-slate-100 text-slate-700 hover:bg-slate-200'
                  }`}
                >
                  <BarChart2 className="w-3.5 h-3.5" /> Confusion Matrix
                </button>
                <button
                  onClick={() => setActiveTab('roc')}
                  className={`text-xs font-extrabold px-3 py-2 rounded-lg flex items-center gap-1.5 transition-all ${
                    activeTab === 'roc'
                      ? darkMode ? 'bg-sky-500 text-slate-950 shadow-md' : 'bg-sky-600 text-white shadow-md'
                      : darkMode ? 'bg-[#0F172A] text-slate-400 hover:text-slate-200' : 'bg-slate-100 text-slate-700 hover:bg-slate-200'
                  }`}
                >
                  <LineChart className="w-3.5 h-3.5" /> ROC Curve
                </button>
                <button
                  onClick={() => setActiveTab('correlation')}
                  className={`text-xs font-extrabold px-3 py-2 rounded-lg flex items-center gap-1.5 transition-all ${
                    activeTab === 'correlation'
                      ? darkMode ? 'bg-sky-500 text-slate-950 shadow-md' : 'bg-sky-600 text-white shadow-md'
                      : darkMode ? 'bg-[#0F172A] text-slate-400 hover:text-slate-200' : 'bg-slate-100 text-slate-700 hover:bg-slate-200'
                  }`}
                >
                  <Activity className="w-3.5 h-3.5" /> Feature Correlation
                </button>
                <button
                  onClick={() => setActiveTab('history')}
                  className={`text-xs font-extrabold px-3 py-2 rounded-lg flex items-center gap-1.5 transition-all ${
                    activeTab === 'history'
                      ? darkMode ? 'bg-sky-500 text-slate-950 shadow-md' : 'bg-sky-600 text-white shadow-md'
                      : darkMode ? 'bg-[#0F172A] text-slate-400 hover:text-slate-200' : 'bg-slate-100 text-slate-700 hover:bg-slate-200'
                  }`}
                >
                  <RefreshCw className="w-3.5 h-3.5" /> Training Curves
                </button>
              </div>

              <div>
                <h3 className={`text-base font-extrabold mb-1 ${darkMode ? 'text-slate-100' : 'text-slate-900'}`}>
                  {graphTabInfo[activeTab].title}
                </h3>
                <p className={`text-xs mb-5 font-medium ${darkMode ? 'text-slate-400' : 'text-slate-600'}`}>
                  {graphTabInfo[activeTab].desc}
                </p>
                <div className={`border rounded-lg p-3 flex justify-center ${
                  darkMode ? 'bg-[#0F172A] border-slate-800' : 'bg-slate-50 border-slate-300'
                }`}>
                  <img 
                    src={graphTabInfo[activeTab].src} 
                    alt={graphTabInfo[activeTab].title}
                    className="max-w-full rounded object-contain shadow-sm"
                    style={{ maxHeight: '420px' }}
                  />
                </div>
              </div>
            </div>
          </div>

          {/* Sidebar / Results Gauge & Info */}
          <div className="space-y-6">
            
            {/* Prediction Result Gauge Card */}
            <div className={`border rounded-xl p-6 shadow-xl transition-all duration-500 ${
              result 
                ? result.prediction === 1 
                  ? darkMode ? 'bg-red-950/30 border-red-800/80' : 'bg-red-50 border-red-300 shadow-red-100'
                  : darkMode ? 'bg-emerald-950/30 border-emerald-800/80' : 'bg-emerald-50 border-emerald-300 shadow-emerald-100'
                : darkMode ? 'bg-[#151F32] border-slate-800' : 'bg-white border-slate-300 shadow-lg shadow-slate-300/30'
            }`}>
              <div className="flex justify-between items-center mb-6">
                <h3 className={`text-sm font-extrabold uppercase tracking-wider ${darkMode ? 'text-slate-100' : 'text-slate-900'}`}>
                  Failure Risk Assessment
                </h3>
                <span className={`text-[10px] font-mono font-bold px-2 py-0.5 rounded border ${
                  darkMode ? 'bg-sky-500/10 text-sky-400 border-sky-500/20' : 'bg-sky-100 text-sky-900 border-sky-300'
                }`}>
                  ANN REALTIME
                </span>
              </div>
              
              {!result && !loading && (
                <div className={`text-center py-8 ${darkMode ? 'text-slate-500' : 'text-slate-500'}`}>
                  <Gauge className="w-12 h-12 mx-auto mb-3 opacity-30 animate-pulse text-sky-500" />
                  <p className="text-xs font-mono font-semibold">Awaiting telemetry inputs...</p>
                </div>
              )}

              {loading && (
                <div className="text-center py-8 text-sky-600 dark:text-sky-400">
                  <RefreshCw className="w-12 h-12 mx-auto mb-3 animate-spin" />
                  <p className="text-xs font-mono font-bold">Processing telemetry through Keras ANN model...</p>
                </div>
              )}

              {result && (
                <div className="space-y-6">
                  <div className="text-center">
                    <p className={`text-[10px] uppercase tracking-widest font-mono font-bold mb-2 ${
                      darkMode ? 'text-slate-400' : 'text-slate-600'
                    }`}>
                      Machine Operational Status
                    </p>
                    <div className={`flex items-center justify-center space-x-2.5 text-2xl font-black tracking-tight ${
                      result.prediction === 1 
                        ? darkMode ? 'text-amber-400' : 'text-amber-600' 
                        : darkMode ? 'text-emerald-400' : 'text-emerald-600'
                    }`}>
                      {result.prediction === 1 ? (
                        <>
                          <AlertTriangle className="w-7 h-7 animate-bounce" />
                          <span>POTENTIAL FAILURE</span>
                        </>
                      ) : (
                        <>
                          <CheckCircle className="w-7 h-7" />
                          <span>NORMAL OPERATION</span>
                        </>
                      )}
                    </div>
                  </div>

                  <div className={`pt-6 border-t ${darkMode ? 'border-slate-800' : 'border-slate-300'}`}>
                    <div className="text-center mb-3">
                       <span className={`text-xs font-mono font-bold uppercase tracking-wider ${
                         darkMode ? 'text-slate-400' : 'text-slate-600'
                       }`}>
                         Failure Probability Score
                       </span>
                       <div className={`text-4xl font-black font-mono mt-1 ${
                         result.prediction === 1 
                           ? darkMode ? 'text-amber-400' : 'text-amber-600' 
                           : darkMode ? 'text-emerald-400' : 'text-emerald-600'
                       }`}>
                         {(result.failure_probability * 100).toFixed(2)}%
                       </div>
                    </div>

                    {/* Industrial Multi-Color Threshold Meter */}
                    <div className={`w-full h-4 rounded-full overflow-hidden shadow-inner border p-0.5 ${
                      darkMode ? 'bg-[#0F172A] border-slate-800' : 'bg-slate-200 border-slate-400'
                    }`}>
                      <div 
                        className={`h-full transition-all duration-1000 ease-out rounded-full ${
                          result.prediction === 1 ? 'bg-amber-500' : 'bg-emerald-500'
                        }`}
                        style={{ width: `${Math.min(100, Math.max(0, result.failure_probability * 100))}%` }}
                      />
                    </div>

                    <div className={`flex justify-between text-[10px] font-mono mt-1 font-semibold ${
                      darkMode ? 'text-slate-500' : 'text-slate-600'
                    }`}>
                      <span>0% (Safe)</span>
                      <span>50% (Threshold)</span>
                      <span>100% (Critical)</span>
                    </div>
                  </div>
                </div>
              )}
            </div>

            {/* Information Panel */}
            <div className={`border rounded-xl p-5 shadow-xl transition-all ${
              darkMode 
                ? 'bg-[#151F32] border-slate-800' 
                : 'bg-white border-slate-300 shadow-lg shadow-slate-300/30'
            }`}>
              <h3 className={`text-sm font-extrabold flex items-center uppercase tracking-wider mb-3 ${
                darkMode ? 'text-slate-100' : 'text-slate-900'
              }`}>
                <Info className="w-4 h-4 mr-2 text-sky-500" />
                Dataset & Architecture
              </h3>
              <p className={`text-xs leading-relaxed mb-4 font-medium ${darkMode ? 'text-slate-400' : 'text-slate-600'}`}>
                Predictive Pulse is trained on the benchmark <strong>AI4I 2020 Predictive Maintenance Dataset</strong> from the UCI Machine Learning Repository (10,000 synthetic operational records).
              </p>
              <button
                onClick={() => setShowTechModal(true)}
                className={`w-full py-2.5 px-3 rounded-lg border text-xs font-bold flex items-center justify-center gap-2 transition-all ${
                  darkMode 
                    ? 'bg-[#0F172A] hover:bg-slate-800 border-slate-700 text-sky-400' 
                    : 'bg-sky-50 hover:bg-sky-100 border-sky-300 text-sky-800'
                }`}
              >
                <FileText className="w-3.5 h-3.5 text-sky-500" />
                View Deep-Dive Technical Specs
              </button>
            </div>

            {/* Evaluation Metrics Card */}
            <div className={`border rounded-xl p-5 shadow-xl transition-all ${
              darkMode 
                ? 'bg-[#151F32] border-slate-800' 
                : 'bg-white border-slate-300 shadow-lg shadow-slate-300/30'
            }`}>
              <h3 className={`text-sm font-extrabold uppercase tracking-wider mb-4 ${
                darkMode ? 'text-slate-100' : 'text-slate-900'
              }`}>
                ANN Model Metrics
              </h3>
              <ul className="space-y-2.5 text-xs font-mono">
                <li className="flex justify-between">
                  <span className={darkMode ? 'text-slate-400' : 'text-slate-600'}>Architecture</span>
                  <span className="font-bold text-sky-600 dark:text-sky-400">Deep Keras ANN</span>
                </li>
                <li className="flex justify-between">
                  <span className={darkMode ? 'text-slate-400' : 'text-slate-600'}>Features</span>
                  <span className="font-bold text-slate-900 dark:text-slate-100">6 Sensors</span>
                </li>
              </ul>
              
              <div className={`mt-4 pt-4 border-t ${darkMode ? 'border-slate-800' : 'border-slate-300'}`}>
                <h4 className={`text-[10px] uppercase tracking-widest font-mono font-bold mb-3 ${
                  darkMode ? 'text-slate-500' : 'text-slate-500'
                }`}>
                  Test Evaluation
                </h4>
                <div className="grid grid-cols-2 gap-3 text-xs font-mono">
                  <div>
                    <p className={darkMode ? 'text-slate-400' : 'text-slate-600'}>Accuracy</p>
                    <p className="font-bold text-emerald-600 dark:text-emerald-400 text-sm">93.10%</p>
                  </div>
                  <div>
                    <p className={darkMode ? 'text-slate-400' : 'text-slate-600'}>ROC-AUC</p>
                    <p className="font-bold text-sky-600 dark:text-sky-400 text-sm">97.16%</p>
                  </div>
                  <div>
                    <p className={darkMode ? 'text-slate-400' : 'text-slate-600'}>Recall</p>
                    <p className="font-bold text-slate-900 dark:text-slate-200">86.76%</p>
                  </div>
                  <div>
                    <p className={darkMode ? 'text-slate-400' : 'text-slate-600'}>Precision</p>
                    <p className="font-bold text-slate-900 dark:text-slate-200">31.38%</p>
                  </div>
                </div>
              </div>
            </div>

          </div>
        </div>

        {/* Technical Architecture & Dataset Modal */}
        {showTechModal && (
          <div className="fixed inset-0 z-50 bg-slate-950/80 backdrop-blur-sm flex items-center justify-center p-4">
            <div className={`max-w-2xl w-full border rounded-2xl p-6 shadow-2xl relative max-h-[85vh] overflow-y-auto ${
              darkMode ? 'bg-[#151F32] border-slate-700 text-slate-200' : 'bg-white border-slate-300 text-slate-800'
            }`}>
              
              <div className="flex justify-between items-center pb-4 mb-4 border-b border-slate-700">
                <div className="flex items-center space-x-3">
                  <div className="p-2 rounded-lg bg-sky-500/10 text-sky-400 border border-sky-500/20">
                    <Code className="w-5 h-5 text-sky-400" />
                  </div>
                  <div>
                    <h3 className={`text-lg font-black ${darkMode ? 'text-slate-100' : 'text-slate-900'}`}>
                      Technical Machine Learning Specifications
                    </h3>
                    <p className="text-xs text-sky-400 font-mono">Dataset Origin & Neural Network Pipeline</p>
                  </div>
                </div>
                
                <button
                  onClick={() => setShowTechModal(false)}
                  className={`p-1.5 rounded-lg border transition-colors ${
                    darkMode ? 'bg-slate-800 hover:bg-slate-700 border-slate-700 text-slate-300' : 'bg-slate-100 hover:bg-slate-200 border-slate-300 text-slate-700'
                  }`}
                >
                  <X className="w-5 h-5" />
                </button>
              </div>

              <div className="space-y-5 text-xs font-sans leading-relaxed">
                
                {/* 1. Dataset Origin */}
                <div className={`p-4 rounded-xl border ${darkMode ? 'bg-[#0F172A] border-slate-800' : 'bg-slate-50 border-slate-200'}`}>
                  <h4 className="font-extrabold text-sm text-sky-400 flex items-center mb-2">
                    <Database className="w-4 h-4 mr-2" /> 1. Dataset Origin & Source
                  </h4>
                  <ul className="space-y-1.5 list-disc list-inside text-slate-300 dark:text-slate-300 text-slate-700 font-medium">
                    <li><strong>Dataset</strong>: AI4I 2020 Predictive Maintenance Benchmark Dataset.</li>
                    <li><strong>Repository</strong>: UCI Machine Learning Repository (<a href="https://archive.ics.uci.edu/ml/datasets/AI4I+2020+Predictive+Maintenance+Dataset" target="_blank" rel="noreferrer" className="text-sky-400 underline">Direct URL</a>).</li>
                    <li><strong>Size</strong>: 10,000 synthetic industrial records reflecting real-world machining failures.</li>
                    <li><strong>Target Variable</strong>: <code className="font-mono bg-sky-500/10 text-sky-400 px-1 py-0.5 rounded">Machine failure</code> (0 = Normal, 1 = Failure).</li>
                  </ul>
                </div>

                {/* 2. Preprocessing Pipeline */}
                <div className={`p-4 rounded-xl border ${darkMode ? 'bg-[#0F172A] border-slate-800' : 'bg-slate-50 border-slate-200'}`}>
                  <h4 className="font-extrabold text-sm text-sky-400 flex items-center mb-2">
                    <Activity className="w-4 h-4 mr-2" /> 2. Preprocessing & Feature Engineering
                  </h4>
                  <ul className="space-y-1.5 list-disc list-inside text-slate-300 dark:text-slate-300 text-slate-700 font-medium">
                    <li><strong>Cleaner</strong>: Dropped row IDs (<code className="font-mono">UDI</code>, <code className="font-mono">Product ID</code>) & leakage columns (<code className="font-mono">TWF</code>, <code className="font-mono">HDF</code>, <code className="font-mono">PWF</code>, <code className="font-mono">OSF</code>, <code className="font-mono">RNF</code>).</li>
                    <li><strong>Standardization</strong>: Continuous numerical values (<code className="font-mono">Air temp</code>, <code className="font-mono">Process temp</code>, <code className="font-mono">Speed</code>, <code className="font-mono">Torque</code>, <code className="font-mono">Tool wear</code>) standardized via <code className="font-mono">StandardScaler</code>.</li>
                    <li><strong>Categorical Encoding</strong>: Quality variant <code className="font-mono">Type</code> (<code className="font-mono">L</code>, <code className="font-mono">M</code>, <code className="font-mono">H</code>) transformed via <code className="font-mono">OneHotEncoder</code>.</li>
                    <li><strong>Data Split</strong>: 80% Training / 20% Testing with Stratified Sampling to handle rare failures.</li>
                  </ul>
                </div>

                {/* 3. Deep Learning Model */}
                <div className={`p-4 rounded-xl border ${darkMode ? 'bg-[#0F172A] border-slate-800' : 'bg-slate-50 border-slate-200'}`}>
                  <h4 className="font-extrabold text-sm text-sky-400 flex items-center mb-2">
                    <Cpu className="w-4 h-4 mr-2" /> 3. Keras Neural Network Architecture
                  </h4>
                  <div className="font-mono bg-slate-950 text-sky-300 p-3 rounded-lg border border-slate-800 text-[11px] mb-2">
                    Input(8 features) → Dense(64, ReLU) → Dropout(0.2) → Dense(32, ReLU) → Dropout(0.2) → Dense(16, ReLU) → Dense(1, Sigmoid)
                  </div>
                  <ul className="space-y-1.5 list-disc list-inside text-slate-300 dark:text-slate-300 text-slate-700 font-medium">
                    <li><strong>Class Weights</strong>: Imbalance handled via <code className="font-mono">compute_class_weight('balanced')</code> during training.</li>
                    <li><strong>Optimizer & Loss</strong>: Trained with <code className="font-mono">Adam(lr=0.001)</code> and <code className="font-mono">binary_crossentropy</code>.</li>
                    <li><strong>Early Stopping</strong>: Monitored validation loss with patience of 15 epochs to prevent overfitting.</li>
                  </ul>
                </div>

                {/* 4. Explainable AI */}
                <div className={`p-4 rounded-xl border ${darkMode ? 'bg-[#0F172A] border-slate-800' : 'bg-slate-50 border-slate-200'}`}>
                  <h4 className="font-extrabold text-sm text-sky-400 flex items-center mb-2">
                    <Layers className="w-4 h-4 mr-2" /> 4. Explainable AI (SHAP)
                  </h4>
                  <p className="text-slate-300 dark:text-slate-300 text-slate-700 font-medium leading-relaxed">
                    Uses <strong>SHAP (SHapley Additive exPlanations)</strong> KernelExplainer to calculate individual Shapley values for each sensor input, enabling full transparency into *why* the neural network predicts failure.
                  </p>
                </div>

              </div>

              <div className="mt-6 pt-4 border-t border-slate-700 flex justify-end">
                <button
                  onClick={() => setShowTechModal(false)}
                  className="bg-sky-500 hover:bg-sky-400 text-slate-950 font-bold py-2 px-5 rounded-lg text-xs uppercase tracking-wider"
                >
                  Close Specs Window
                </button>
              </div>

            </div>
          </div>
        )}

      </div>
    </div>
  );
};

export default App;
