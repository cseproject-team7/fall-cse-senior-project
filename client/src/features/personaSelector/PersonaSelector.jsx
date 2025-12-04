import React from 'react';
import { User } from 'lucide-react';
import { formatPersonaName } from '../../utils/dateFormatter';

const PersonaSelector = ({ personas, selectedPersona, onPersonaChange, disabled }) => {
  return (
    <div className="bg-white p-1 rounded-xl shadow-sm border border-gray-200 flex items-center">
      <div className="px-3 text-gray-500">
        <User className="w-5 h-5" />
      </div>
      <select
        value={selectedPersona}
        onChange={(e) => onPersonaChange(e.target.value)}
        disabled={disabled}
        className="bg-transparent py-2 pr-8 pl-2 text-sm font-medium text-gray-700 focus:outline-none cursor-pointer hover:text-[#006747] transition-colors"
      >
        {personas.map((persona) => (
          <option key={persona} value={persona}>
            {formatPersonaName(persona)}
          </option>
        ))}
      </select>
    </div>
  );
};

export default PersonaSelector;
