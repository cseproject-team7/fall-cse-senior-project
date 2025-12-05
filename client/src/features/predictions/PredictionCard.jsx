import React, { useState } from 'react';
import { CheckCircle2, AlertCircle, Plus } from 'lucide-react';

const PredictionCard = ({ 
  prediction, 
  mode, // 'auto' or 'manual'
  onIncorrectPrediction,
  onAddTestApp
}) => {
  const [customApp, setCustomApp] = useState('');

  if (!prediction.next_pattern) return null;

  const isManual = mode === 'manual';

  return (
    <div className={`bg-gradient-to-br ${isManual ? 'from-purple-600 to-purple-800' : 'from-[#006747] to-[#004d34]'} p-6 rounded-xl shadow-lg text-white relative overflow-hidden`}>
      <div className="absolute top-0 right-0 w-40 h-40 bg-white opacity-5 rounded-bl-full -mr-10 -mt-10"></div>

      <div className="relative z-10">
        <div className="flex items-center gap-2 mb-4">
          <CheckCircle2 className="w-5 h-5" />
          <h3 className="text-sm uppercase tracking-wider font-bold opacity-90">
            {isManual ? 'Manual Test Prediction' : 'Auto Prediction'}
          </h3>
        </div>
        <div className="text-3xl font-extrabold mb-4">{prediction.next_pattern}</div>

        <div className="bg-white bg-opacity-10 rounded-lg p-4 backdrop-blur-sm">
          <div className="flex items-center justify-between mb-2">
            <span className="text-sm font-semibold">Confidence</span>
            <span className="text-lg font-bold text-[#CDB87D]">
              {(prediction.pattern_confidence * 100).toFixed(1)}%
            </span>
          </div>
          <div className="w-full bg-white bg-opacity-20 rounded-full h-2">
            <div
              className="bg-[#CDB87D] h-2 rounded-full transition-all duration-500"
              style={{ width: `${prediction.pattern_confidence * 100}%` }}
            ></div>
          </div>

          {!isManual && (
            <button
              onClick={onIncorrectPrediction}
              className="mt-4 w-full bg-white bg-opacity-20 hover:bg-opacity-30 text-white px-4 py-2 rounded-lg text-sm font-semibold transition-all flex items-center justify-center gap-2"
            >
              <AlertCircle className="w-4 h-4" />
              Incorrect Prediction
            </button>
          )}
        </div>

        {prediction.next_apps && prediction.next_apps.length > 0 && (
          <div className="mt-4">
            <h4 className="text-xs uppercase tracking-wider font-semibold opacity-80 mb-3">
              Predicted App Sequence
            </h4>
            
            <div className="space-y-2">
              {prediction.next_apps.slice(0, 5).map((app, idx) => (
                <div
                  key={idx}
                  className="bg-white bg-opacity-10 backdrop-blur-sm px-3 py-2 rounded-lg text-xs font-medium flex items-center gap-2"
                >
                  <span className="opacity-60">{idx + 1}.</span>
                  <span>{app}</span>
                </div>
              ))}
            </div>
          </div>
        )}

        {isManual && (
          <div className="mt-4 pt-4 border-t border-white border-opacity-20">
            <h4 className="text-xs uppercase tracking-wider font-semibold opacity-80 mb-2">
              Add Test App
            </h4>
            <div className="flex gap-2">
              <input
                type="text"
                value={customApp}
                onChange={(e) => setCustomApp(e.target.value)}
                placeholder="e.g., Outlook, Canvas"
                className="flex-1 px-3 py-2 rounded-lg text-sm bg-white bg-opacity-20 text-white placeholder-white placeholder-opacity-50 focus:outline-none focus:ring-2 focus:ring-white focus:ring-opacity-50"
              />
              <button
                onClick={() => {
                  if (customApp.trim()) {
                    onAddTestApp(customApp.trim());
                    setCustomApp('');
                  }
                }}
                className="bg-white bg-opacity-20 hover:bg-opacity-30 px-4 py-2 rounded-lg flex items-center gap-2 transition-all"
              >
                <Plus className="w-4 h-4" />
                Add
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default PredictionCard;
