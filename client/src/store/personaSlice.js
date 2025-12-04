import { createSlice, createAsyncThunk } from '@reduxjs/toolkit';
import { logsApi } from '../api/logsApi';

export const fetchPersonas = createAsyncThunk(
  'persona/fetchPersonas',
  async () => {
    const response = await logsApi.getLogsGroupedByUser();
    return response.users; // Return array of user names
  }
);

const personaSlice = createSlice({
  name: 'persona',
  initialState: {
    personas: [],
    selectedPersona: '',
    loading: false,
    error: null,
  },
  reducers: {
    setSelectedPersona: (state, action) => {
      state.selectedPersona = action.payload;
    },
  },
  extraReducers: (builder) => {
    builder
      .addCase(fetchPersonas.pending, (state) => {
        state.loading = true;
        state.error = null;
      })
      .addCase(fetchPersonas.fulfilled, (state, action) => {
        state.loading = false;
        state.personas = action.payload;
        if (action.payload.length > 0 && !state.selectedPersona) {
          state.selectedPersona = action.payload[0];
        }
      })
      .addCase(fetchPersonas.rejected, (state, action) => {
        state.loading = false;
        state.error = action.error.message;
      });
  },
});

export const { setSelectedPersona } = personaSlice.actions;
export default personaSlice.reducer;
