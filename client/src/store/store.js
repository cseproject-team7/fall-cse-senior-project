import { configureStore } from '@reduxjs/toolkit';
import personaReducer from './personaSlice';
import logsReducer from './logsSlice';
import predictionsReducer from './predictionsSlice';

export const store = configureStore({
  reducer: {
    persona: personaReducer,
    logs: logsReducer,
    predictions: predictionsReducer,
  },
});
