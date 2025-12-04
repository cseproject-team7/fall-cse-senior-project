import React from 'react';
import { BrainCircuit } from 'lucide-react';
import PredictionCard from './PredictionCard';
import PredictionTimeline from './PredictionTimeline';

const Predictions = ({ 
  predictions, 
  loading, 
  onIncorrectPrediction, 
  onRecordAppAccess, 
  recordingApp 
}) => {
  return (
    <div className="bg-white rounded-xl shadow-md border border-gray-200 flex flex-col h-full overflow-hidden">
      <div className="p-6 border-b border-gray-100 flex justify-between items-center bg-gray-50 rounded-t-xl flex-shrink-0">
        <div className="flex items-center gap-2">
          <BrainCircuit className="w-5 h-5 text-[#CDB87D]" />
          <h2 className="text-lg font-bold text-gray-800">ML Predictions</h2>
        </div>
        <span className="px-3 py-1 bg-[#CDB87D] text-[#006747] text-xs font-bold rounded-full">
          {predictions.predictions?.length || 0} Prediction
          {predictions.predictions?.length !== 1 ? 's' : ''}
        </span>
      </div>

      <div className="flex-1 overflow-y-auto p-6 bg-[#fcfcfc]">
        {loading ? (
          <div className="flex flex-col items-center justify-center h-full">
            <div className="loader"></div>
            <p className="mt-4 text-gray-500">Generating predictions...</p>
          </div>
        ) : !predictions.predictions || predictions.predictions.length === 0 ? (
          <div className="flex flex-col items-center justify-center h-full text-gray-400">
            <BrainCircuit className="w-16 h-16 mb-4 opacity-10" />
            <p>No predictions available</p>
          </div>
        ) : (
          <div className="space-y-4">
            <PredictionCard
              prediction={predictions}
              onIncorrectPrediction={onIncorrectPrediction}
              onRecordAppAccess={onRecordAppAccess}
              recordingApp={recordingApp}
            />
            <PredictionTimeline predictions={predictions.predictions} />
          </div>
        )}
      </div>
    </div>
  );
};

export default Predictions;
