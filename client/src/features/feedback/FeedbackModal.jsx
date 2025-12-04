import React, { useState } from 'react';
import { AlertCircle } from 'lucide-react';

const FeedbackModal = ({ onSubmit, onClose }) => {
  const [feedbackApp, setFeedbackApp] = useState('');
  const [error, setError] = useState('');

  const handleSubmit = () => {
    if (!feedbackApp.trim()) {
      setError('Please enter the actual app you accessed');
      return;
    }
    onSubmit(feedbackApp);
    setFeedbackApp('');
    setError('');
  };

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
      <div className="bg-white rounded-xl shadow-2xl p-8 max-w-md w-full mx-4">
        <div className="flex items-center gap-3 mb-6">
          <div className="w-12 h-12 bg-red-100 rounded-full flex items-center justify-center">
            <AlertCircle className="w-6 h-6 text-red-600" />
          </div>
          <div>
            <h3 className="text-xl font-bold text-gray-900">Incorrect Prediction</h3>
            <p className="text-sm text-gray-500">Help us improve the model</p>
          </div>
        </div>

        <div className="mb-6">
          <label className="block text-sm font-semibold text-gray-700 mb-2">
            What app did you actually access?
          </label>
          <input
            type="text"
            value={feedbackApp}
            onChange={(e) => {
              setFeedbackApp(e.target.value);
              setError('');
            }}
            placeholder="Enter app name (e.g., Canvas, Teams)"
            className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-[#006747] focus:border-transparent outline-none"
            autoFocus
          />
          {error && <p className="text-xs text-red-600 mt-2">{error}</p>}
          <p className="text-xs text-gray-500 mt-2">
            This data will be stored for model retraining via Kafka
          </p>
        </div>

        <div className="flex gap-3">
          <button
            onClick={() => {
              onClose();
              setFeedbackApp('');
              setError('');
            }}
            className="flex-1 px-4 py-2 border border-gray-300 rounded-lg text-gray-700 font-semibold hover:bg-gray-50 transition-colors"
          >
            Cancel
          </button>
          <button
            onClick={handleSubmit}
            className="flex-1 px-4 py-2 bg-[#006747] text-white rounded-lg font-semibold hover:bg-[#004d34] transition-colors"
          >
            Submit Feedback
          </button>
        </div>
      </div>
    </div>
  );
};

export default FeedbackModal;
