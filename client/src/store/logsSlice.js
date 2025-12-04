import { createSlice, createAsyncThunk } from '@reduxjs/toolkit';
import { logsApi } from '../api/logsApi';

export const fetchLogs = createAsyncThunk(
  'logs/fetchLogs',
  async (persona) => {
    const response = await logsApi.getLogsByPersona(persona);
    return response.logs;
  }
);

const logsSlice = createSlice({
  name: 'logs',
  initialState: {
    logs: [],
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
