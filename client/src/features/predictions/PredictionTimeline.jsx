import React from 'react';
import { Activity } from 'lucide-react';

const PredictionTimeline = ({ predictions }) => {
  if (!predictions || predictions.length === 0) return null;

  return (
    <div className="bg-white p-6 rounded-xl border border-gray-200">
      <h3 className="text-sm font-bold text-gray-700 mb-4 flex items-center gap-2">
        <Activity className="w-4 h-4 text-[#006747]" />
        Future Pattern Sequence
      </h3>
      <div className="space-y-2">
        {predictions.slice(0, 10).map((pred, index) => (
          <div
            key={index}
            className="flex items-center gap-3 p-3 bg-gray-50 rounded-lg hover:bg-[#fdfcf6] transition-colors group"
          >
            <div className="flex-shrink-0 w-8 h-8 rounded-full bg-[#006747] text-white flex items-center justify-center text-xs font-bold">
              {pred.step}
            </div>
            <div className="flex-1">
              <div className="font-semibold text-gray-900 text-sm">{pred.pattern}</div>
              {pred.top_apps && pred.top_apps.length > 0 && (
                <div className="text-xs text-gray-500 mt-1">
                  {pred.top_apps.slice(0, 3).join(', ')}
                </div>
              )}
            </div>
            <div className="flex-shrink-0">
              <div className="text-xs font-semibold text-gray-400">
                {(pred.confidence * 100).toFixed(0)}%
              </div>
              <div className="w-16 bg-gray-200 rounded-full h-1.5 mt-1">
                <div
                  className="bg-[#006747] h-1.5 rounded-full"
                  style={{ width: `${pred.confidence * 100}%` }}
                ></div>
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

export default PredictionTimeline;
