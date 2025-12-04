import React from 'react';
import { CheckCircle2, AlertCircle } from 'lucide-react';

const PredictionCard = ({ 
  prediction, 
  onIncorrectPrediction, 
  onRecordAppAccess, 
  recordingApp 
}) => {
  if (!prediction.next_pattern) return null;

  return (
    <div className="bg-gradient-to-br from-[#006747] to-[#004d34] p-6 rounded-xl shadow-lg text-white relative overflow-hidden">
      <div className="absolute top-0 right-0 w-40 h-40 bg-white opacity-5 rounded-bl-full -mr-10 -mt-10"></div>

      <div className="relative z-10">
        <div className="flex items-center gap-2 mb-4">
          <CheckCircle2 className="w-5 h-5" />
          <h3 className="text-sm uppercase tracking-wider font-bold opacity-90">
            Next Predicted Pattern
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

          <button
            onClick={onIncorrectPrediction}
            className="mt-4 w-full bg-white bg-opacity-20 hover:bg-opacity-30 text-white px-4 py-2 rounded-lg text-sm font-semibold transition-all flex items-center justify-center gap-2"
          >
            <AlertCircle className="w-4 h-4" />
            Incorrect Prediction
          </button>
        </div>

        {prediction.next_apps && prediction.next_apps.length > 0 && (
          <div className="mt-4">
            <h4 className="text-xs uppercase tracking-wider font-semibold opacity-80 mb-2">
              Expected Apps
            </h4>
            <div className="flex flex-wrap gap-2">
              {prediction.next_apps.slice(0, 5).map((app, idx) => (
                <button
                  key={idx}
                  onClick={() => onRecordAppAccess(app)}
                  disabled={recordingApp === app}
                  className="bg-white bg-opacity-20 backdrop-blur-sm px-3 py-1 rounded-full text-xs font-medium hover:bg-opacity-30 transition-all cursor-pointer disabled:opacity-50 disabled:cursor-wait"
                  title="Click to record this app access"
                >
                  {recordingApp === app ? '⏳ Recording...' : `✓ ${app}`}
                </button>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default PredictionCard;
