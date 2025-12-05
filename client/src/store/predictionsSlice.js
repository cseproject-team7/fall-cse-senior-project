import { createSlice, createAsyncThunk } from '@reduxjs/toolkit';
import { predictionsApi } from '../api/predictionsApi';

// Async thunk to generate automatic predictions (real logs only)
export const generateAutoPredictions = createAsyncThunk(
  'predictions/generateAuto',
  async (logs) => {
    const response = await predictionsApi.predict(logs);
    return response.prediction;
  }
);

// Async thunk to generate manual predictions (with test apps)
export const generateManualPredictions = createAsyncThunk(
  'predictions/generateManual',
  async (logs) => {
    const response = await predictionsApi.predict(logs);
    return response.prediction;
  }
);

const predictionsSlice = createSlice({
  name: 'predictions',
  initialState: {
    autoPredictions: {},
    manualPredictions: {},
    manualLogs: [], // User's test logs
    loading: false,
    error: null,
  },
  reducers: {
    addManualTestApp: (state, action) => {
      const { appDisplayName, timestamp } = action.payload;
      state.manualLogs.push({
        appDisplayName,
        createdDateTime: timestamp || new Date().toISOString(),
        isManualTest: true
      });
    },
    resetManualLogs: (state, action) => {
      state.manualLogs = action.payload; // Reset to original logs
    },
  },
  extraReducers: (builder) => {
    builder
      // Auto predictions
      .addCase(generateAutoPredictions.pending, (state) => {
        state.loading = true;
        state.error = null;
      })
      .addCase(generateAutoPredictions.fulfilled, (state, action) => {
        state.loading = false;
        state.autoPredictions = action.payload;
      })
      .addCase(generateAutoPredictions.rejected, (state, action) => {
        state.loading = false;
        state.error = action.error.message;
      })
      // Manual predictions
      .addCase(generateManualPredictions.pending, (state) => {
        state.loading = true;
        state.error = null;
      })
      .addCase(generateManualPredictions.fulfilled, (state, action) => {
        state.loading = false;
        state.manualPredictions = action.payload;
      })
      .addCase(generateManualPredictions.rejected, (state, action) => {
        state.loading = false;
        state.error = action.error.message;
      });
  },
});

export const { addManualTestApp, resetManualLogs } = predictionsSlice.actions;
export default predictionsSlice.reducer;
