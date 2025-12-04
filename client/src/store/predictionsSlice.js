import { createSlice, createAsyncThunk } from '@reduxjs/toolkit';
import { predictionsApi } from '../api/predictionsApi';

export const generatePredictions = createAsyncThunk(
  'predictions/generate',
  async (logs) => {
    const response = await predictionsApi.predict(logs);
    return response.prediction;
  }
);

export const recordAppAccess = createAsyncThunk(
  'predictions/recordAppAccess',
  async ({ logs, appDisplayName }) => {
    const response = await predictionsApi.recordAppAccess(logs, appDisplayName);
    return response.prediction;
  }
);

const predictionsSlice = createSlice({
  name: 'predictions',
  initialState: {
    predictions: {},
    loading: false,
    error: null,
    recordingApp: null,
  },
  reducers: {
    setRecordingApp: (state, action) => {
      state.recordingApp = action.payload;
    },
  },
  extraReducers: (builder) => {
    builder
      .addCase(generatePredictions.pending, (state) => {
        state.loading = true;
        state.error = null;
      })
      .addCase(generatePredictions.fulfilled, (state, action) => {
        state.loading = false;
        state.predictions = action.payload;
      })
      .addCase(generatePredictions.rejected, (state, action) => {
        state.loading = false;
        state.error = action.error.message;
      })
      .addCase(recordAppAccess.pending, (state, action) => {
        state.recordingApp = action.meta.arg.appDisplayName;
      })
      .addCase(recordAppAccess.fulfilled, (state, action) => {
        state.predictions = action.payload;
        state.recordingApp = null;
      })
      .addCase(recordAppAccess.rejected, (state, action) => {
        state.error = action.error.message;
        state.recordingApp = null;
      });
  },
});

export const { setRecordingApp } = predictionsSlice.actions;
export default predictionsSlice.reducer;
