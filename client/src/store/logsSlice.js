import { createSlice, createAsyncThunk } from '@reduxjs/toolkit';
import { logsApi } from '../api/logsApi';

// Fetch all grouped logs once
export const fetchAllGroupedLogs = createAsyncThunk(
  'logs/fetchAllGroupedLogs',
  async () => {
    const response = await logsApi.getLogsGroupedByUser();
    return response; // Returns { users: [...], logsByUser: {...} }
  }
);

// Action to select logs for a specific user
export const fetchLogs = createAsyncThunk(
  'logs/fetchLogs',
  async (persona, { getState }) => {
    const state = getState();
    
    // If we already have grouped logs, use them
    if (state.logs.logsByUser && state.logs.logsByUser[persona]) {
      return state.logs.logsByUser[persona];
    }
    
    // Otherwise fetch grouped logs
    const response = await logsApi.getLogsGroupedByUser();
    return response.logsByUser[persona] || [];
  }
);

const logsSlice = createSlice({
  name: 'logs',
  initialState: {
    logs: [],
    logsByUser: null, // Store all grouped logs
    loading: false,
    error: null,
  },
  reducers: {
    addLog: (state, action) => {
      state.logs.push(action.payload);
    },
  },
  extraReducers: (builder) => {
    builder
      .addCase(fetchAllGroupedLogs.pending, (state) => {
        state.loading = true;
        state.error = null;
      })
      .addCase(fetchAllGroupedLogs.fulfilled, (state, action) => {
        state.loading = false;
        state.logsByUser = action.payload.logsByUser;
      })
      .addCase(fetchAllGroupedLogs.rejected, (state, action) => {
        state.loading = false;
        state.error = action.error.message;
      })
      .addCase(fetchLogs.pending, (state) => {
        state.loading = true;
        state.error = null;
      })
      .addCase(fetchLogs.fulfilled, (state, action) => {
        state.loading = false;
        state.logs = action.payload;
      })
      .addCase(fetchLogs.rejected, (state, action) => {
        state.loading = false;
        state.error = action.error.message;
      });
  },
});

export const { addLog } = logsSlice.actions;
export default logsSlice.reducer;
