import React from 'react';
import { List, Activity, Calendar, Clock } from 'lucide-react';
import { formatHour, formatWeekday } from '../../utils/dateFormatter';

const ActivityLogs = ({ logs, loading }) => {
  return (
    <div className="bg-white rounded-xl shadow-md border border-gray-200 flex flex-col h-full overflow-hidden">
      <div className="p-6 border-b border-gray-100 flex justify-between items-center bg-gray-50 rounded-t-xl">
        <div className="flex items-center gap-2">
          <List className="w-5 h-5 text-gray-500" />
          <h2 className="text-lg font-bold text-gray-800">Input Activity Logs</h2>
        </div>
        <span className="px-3 py-1 bg-[#006747] text-white text-xs font-bold rounded-full">
          {logs.length} Logs
        </span>
      </div>

      <div className="flex-1 overflow-y-auto p-0">
        {loading ? (
          <div className="flex flex-col items-center justify-center h-full">
            <div className="loader"></div>
            <p className="mt-4 text-gray-500">Analyzing patterns...</p>
          </div>
        ) : logs.length === 0 ? (
          <div className="flex flex-col items-center justify-center h-full text-gray-400">
            <List className="w-12 h-12 mb-2 opacity-20" />
            <p>No logs available for this persona</p>
          </div>
        ) : (
          <div className="divide-y divide-gray-100">
            {logs.map((log, index) => (
              <div
                key={index}
                className="p-4 hover:bg-[#fdfcf6] transition-colors flex items-center justify-between group"
              >
                <div className="flex items-center gap-4">
                  <div className="w-10 h-10 rounded-full bg-gray-100 flex items-center justify-center text-gray-500 group-hover:bg-[#EDEBD1] group-hover:text-[#006747] transition-colors">
                    <Activity className="w-5 h-5" />
                  </div>
                  <div>
                    <h3 className="font-bold text-gray-800">{log.appDisplayName}</h3>
                    <div className="flex items-center gap-3 text-xs text-gray-500 mt-1">
                      <span className="flex items-center gap-1">
                        <Calendar className="w-3 h-3" />
                        {formatWeekday(log.weekday)}
                      </span>
                      <span className="flex items-center gap-1">
                        <Clock className="w-3 h-3" />
                        {formatHour(log.hour)}
                      </span>
                    </div>
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
};

export default ActivityLogs;
