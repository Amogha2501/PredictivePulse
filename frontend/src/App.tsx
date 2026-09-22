import React, { useState } from 'react';
import { Activity, AlertTriangle, CheckCircle, Info, RefreshCw, BarChart2, Database, Download } from 'lucide-react';
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

const App: React.FC = () => {
  const [formData, setFormData] = useState<MachineDataInput>(DEFAULT_STATE);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<PredictionResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loadedSampleInfo, setLoadedSampleInfo] = useState<string | null>(null);

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
    setLoadedSampleInfo("Normal sample loaded from test dataset");
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
    setLoadedSampleInfo("Failure sample loaded from test dataset (PWF)");
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setResult(null);

    // Validate inputs aren't empty string
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
      setError(err.message || 'An unexpected error occurred.');
    } finally {
      setLoading(false);
    }
  };

  // Helper to render input field with validation
  const renderInput = (label: string, name: keyof typeof RANGES, step: string) => {
    const val = formData[name];
    const isOutOfRange = 
      typeof val === 'number' && 
      (val < RANGES[name].min || val > RANGES[name].max);

    return (
      <div className="space-y-1 relative">
        <label className="text-sm font-medium text-slate-400 flex justify-between">
          {label}
        </label>
        <div className="relative">
          <input
            type="number"
            step={step}
            name={name}
            value={val}
            onChange={handleInputChange}
            className={`w-full bg-slate-950 border ${isOutOfRange ? 'border-amber-500/50 focus:border-amber-500 focus:ring-amber-500' : 'border-slate-800 focus:border-cyan-500 focus:ring-cyan-500'} rounded-lg px-4 py-2.5 focus:outline-none focus:ring-1 transition-colors`}
            required
          />
        </div>
        <div className="flex justify-between items-start mt-1">
          <span className="text-xs text-slate-500">
            Typical range: {RANGES[name].min} – {RANGES[name].max}
          </span>
          {isOutOfRange && (
            <span className="text-[10px] text-amber-500/90 flex items-center ml-2 bg-amber-500/10 px-1.5 py-0.5 rounded">
              <AlertTriangle className="w-3 h-3 mr-1" />
              Outside training range
            </span>
          )}
        </div>
      </div>
    );
  };

  return (
    <div className="min-h-screen bg-slate-950 text-slate-200 p-4 md:p-8 font-sans">
      <div className="max-w-6xl mx-auto space-y-8">
        
        {/* Header */}
        <header className="flex flex-col sm:flex-row sm:items-center justify-between pb-6 border-b border-slate-800 gap-4">
          <div className="flex items-center space-x-4">
            <Activity className="w-10 h-10 text-cyan-500" />
            <div>
              <h1 className="text-3xl font-bold text-slate-100 tracking-tight">Predictive Pulse</h1>
              <p className="text-slate-400">An Explainable Deep Learning Framework for Predictive Maintenance</p>
            </div>
          </div>
          <div className="flex gap-2 shrink-0">
            <button onClick={loadNormalSample} className="text-xs bg-slate-800 hover:bg-slate-700 text-slate-300 px-3 py-2 rounded flex items-center transition-colors">
              <Download className="w-3 h-3 mr-1.5" /> Normal Sample
            </button>
            <button onClick={loadFailureSample} className="text-xs bg-slate-800 hover:bg-slate-700 text-slate-300 px-3 py-2 rounded flex items-center transition-colors">
              <Download className="w-3 h-3 mr-1.5" /> Failure Sample
            </button>
          </div>
        </header>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          
          {/* Main Input Form Section */}
          <div className="lg:col-span-2 space-y-8">
            <div className="bg-slate-900 border border-slate-800 rounded-xl p-6 shadow-xl relative">
              
              <div className="flex justify-between items-center mb-6">
                <h2 className="text-xl font-semibold flex items-center text-slate-100">
                  <BarChart2 className="w-5 h-5 mr-2 text-cyan-500" />
                  Machine Operating Parameters
                </h2>
                {loadedSampleInfo && (
                  <span className="text-xs bg-cyan-900/40 text-cyan-400 border border-cyan-800/50 px-2 py-1 rounded animate-pulse">
                    {loadedSampleInfo}
                  </span>
                )}
              </div>
              
              <form onSubmit={handleSubmit} className="space-y-6">
                <div className="grid grid-cols-1 md:grid-cols-2 gap-x-6 gap-y-5">
                  
                  {/* Type */}
                  <div className="space-y-1">
                    <label className="text-sm font-medium text-slate-400">Product Quality Type</label>
                    <select
                      name="Type"
                      value={formData['Type']}
                      onChange={handleInputChange}
                      className="w-full bg-slate-950 border border-slate-800 rounded-lg px-4 py-2.5 focus:outline-none focus:border-cyan-500 focus:ring-1 focus:ring-cyan-500 transition-colors"
                    >
                      <option value="L">Low (L)</option>
                      <option value="M">Medium (M)</option>
                      <option value="H">High (H)</option>
                    </select>
                    <div className="text-xs text-slate-500 mt-1 h-4">Categorical product variant</div>
                  </div>

                  {renderInput("Air Temperature [K]", "Air temperature [K]", "0.1")}
                  {renderInput("Process Temperature [K]", "Process temperature [K]", "0.1")}
                  {renderInput("Rotational Speed [rpm]", "Rotational speed [rpm]", "1")}
                  {renderInput("Torque [Nm]", "Torque [Nm]", "0.1")}
                  {renderInput("Tool Wear [min]", "Tool wear [min]", "1")}
                  
                </div>

                <div className="flex space-x-4 pt-4 border-t border-slate-800/50 mt-6">
                  <button
                    type="submit"
                    disabled={loading}
                    className="flex-1 bg-cyan-600 hover:bg-cyan-500 text-white font-semibold py-3 px-6 rounded-lg transition-colors flex justify-center items-center disabled:opacity-50 disabled:cursor-not-allowed"
                  >
                    {loading ? (
                      <>
                        <RefreshCw className="w-5 h-5 mr-2 animate-spin" />
                        Analyzing...
                      </>
                    ) : (
                      'PREDICT FAILURE'
                    )}
                  </button>
                  <button
                    type="button"
                    onClick={handleReset}
                    className="bg-slate-800 hover:bg-slate-700 text-slate-300 font-semibold py-3 px-6 rounded-lg transition-colors"
                  >
                    Reset
                  </button>
                </div>
              </form>

              {/* Error Message */}
              {error && (
                <div className="mt-6 bg-red-900/30 border border-red-800 text-red-200 p-4 rounded-lg flex items-start">
                  <AlertTriangle className="w-5 h-5 mr-3 shrink-0 mt-0.5" />
                  <div>
                    <h4 className="font-semibold text-red-100">Prediction Failed</h4>
                    <p className="text-sm mt-1">{error}</p>
                  </div>
                </div>
              )}
            </div>

            {/* Explainability Section */}
            <div className="bg-slate-900 border border-slate-800 rounded-xl p-6 shadow-xl">
              <h2 className="text-xl font-semibold mb-2 text-slate-100">Global SHAP Feature Importance — Test Set</h2>
              <p className="text-slate-400 text-sm mb-6">
                The SHAP summary plot shows how the input features influence the model's predictions across the test set.
                <br/>
                <span className="text-slate-500 text-xs italic block mt-2">Note: Global explanation based on the model's test-set predictions.</span>
              </p>
              <div className="bg-slate-950 border border-slate-800 rounded-lg p-2 flex justify-center">
                <img 
                  src="/shap_summary.png" 
                  alt="SHAP Summary Plot" 
                  className="max-w-full rounded-md object-contain"
                  style={{ maxHeight: '400px' }}
                />
              </div>
            </div>
          </div>

          {/* Sidebar / Results & Info */}
          <div className="space-y-8">
            
            {/* Prediction Result Card */}
            <div className={`border rounded-xl p-6 shadow-xl transition-all duration-500 ${
              result 
                ? result.prediction === 1 
                  ? 'bg-red-900/20 border-red-800' 
                  : 'bg-emerald-900/20 border-emerald-800'
                : 'bg-slate-900 border-slate-800'
            }`}>
              <h2 className="text-xl font-semibold mb-6 text-slate-100">Prediction Result</h2>
              
              {!result && !loading && (
                <div className="text-center py-8 text-slate-500">
                  <Activity className="w-12 h-12 mx-auto mb-3 opacity-30" />
                  <p>Awaiting machine parameters...</p>
                </div>
              )}

              {loading && (
                <div className="text-center py-8 text-cyan-500">
                  <RefreshCw className="w-12 h-12 mx-auto mb-3 animate-spin" />
                  <p>Processing via ANN...</p>
                </div>
              )}

              {result && (
                <div className="space-y-6">
                  <div className="text-center">
                    <p className="text-slate-400 text-sm uppercase tracking-wider font-semibold mb-3">Status</p>
                    <div className={`flex items-center justify-center space-x-3 text-3xl font-bold tracking-tight ${
                      result.prediction === 1 ? 'text-red-400' : 'text-emerald-400'
                    }`}>
                      {result.prediction === 1 ? (
                        <>
                          <AlertTriangle className="w-8 h-8" />
                          <span>Potential Failure</span>
                        </>
                      ) : (
                        <>
                          <CheckCircle className="w-8 h-8" />
                          <span>Normal</span>
                        </>
                      )}
                    </div>
                  </div>

                  <div className="pt-6 border-t border-slate-800/50">
                    <div className="text-center mb-4">
                       <span className="text-slate-400 text-sm font-medium">Predicted probability of machine failure:</span>
                       <div className="text-2xl font-bold text-slate-100 mt-1">
                         {(result.failure_probability * 100).toFixed(2)}%
                       </div>
                    </div>
                    {/* Progress Bar */}
                    <div className="w-full h-3 bg-slate-950 rounded-full overflow-hidden shadow-inner border border-slate-800/50">
                      <div 
                        className={`h-full transition-all duration-1000 ease-out rounded-full ${
                          result.prediction === 1 ? 'bg-red-500' : 'bg-emerald-500'
                        }`}
                        style={{ width: `${Math.min(100, Math.max(0, result.failure_probability * 100))}%` }}
                      />
                    </div>
                  </div>
                </div>
              )}
            </div>

            {/* Information Panels */}
            <div className="bg-slate-900 border border-slate-800 rounded-xl p-6 shadow-xl">
              <h3 className="text-lg font-semibold flex items-center text-slate-100 mb-4">
                <Info className="w-5 h-5 mr-2 text-cyan-500" />
                What is Predictive Pulse?
              </h3>
              <p className="text-slate-400 text-sm leading-relaxed mb-4">
                Predictive Pulse analyzes machine operating parameters using a deep learning model to estimate the likelihood of machine failure and uses Explainable AI to understand the factors influencing predictions.
              </p>
              <div className="flex items-start bg-slate-950 p-3 rounded-lg border border-slate-800/80">
                <Database className="w-4 h-4 mr-2 text-slate-500 shrink-0 mt-0.5" />
                <p className="text-xs text-slate-500 leading-relaxed">
                  <strong className="text-slate-300 font-medium">Dataset: AI4I 2020 Predictive Maintenance Dataset</strong><br/>
                  The AI4I 2020 dataset is a synthetic benchmark dataset designed for predictive-maintenance research.
                </p>
              </div>
            </div>

            <div className="bg-slate-900 border border-slate-800 rounded-xl p-6 shadow-xl">
              <h3 className="text-lg font-semibold text-slate-100 mb-4">About the Model</h3>
              <ul className="space-y-3 text-sm text-slate-300">
                <li className="flex justify-between">
                  <span className="text-slate-500">Model</span>
                  <span className="font-medium">Artificial Neural Network (ANN)</span>
                </li>
                <li className="flex justify-between">
                  <span className="text-slate-500">Input Features</span>
                  <span className="font-medium">6</span>
                </li>
              </ul>
              
              <div className="mt-5 pt-5 border-t border-slate-800">
                <h4 className="text-xs uppercase tracking-wider text-slate-500 font-semibold mb-3">Test-Set Evaluation Results</h4>
                <div className="grid grid-cols-2 gap-4 text-sm">
                  <div>
                    <p className="text-slate-500">Accuracy</p>
                    <p className="font-medium text-slate-200">93.10%</p>
                  </div>
                  <div>
                    <p className="text-slate-500">ROC-AUC</p>
                    <p className="font-medium text-slate-200">97.16%</p>
                  </div>
                  <div>
                    <p className="text-slate-500">Recall</p>
                    <p className="font-medium text-slate-200">86.76%</p>
                  </div>
                  <div>
                    <p className="text-slate-500">Precision</p>
                    <p className="font-medium text-slate-200">31.38%</p>
                  </div>
                  <div>
                    <p className="text-slate-500">F1-score</p>
                    <p className="font-medium text-slate-200">46.09%</p>
                  </div>
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
