import { createSlice, PayloadAction, createAsyncThunk } from '@reduxjs/toolkit';
import { requestAxios } from '@static/function';
import { MANAGER_IP } from '@static/const';
// Define a type for the slice state
interface ExperimentState {
    status: string;
    error: string;
}

// Define the initial state using that type
const initialState: ExperimentState = {
    status: '',
    error: ''
};

export const getExperimentstatus = createAsyncThunk(
    'test/fetchIp', // 这个名字有什么意义呢？
    async () => {
        const response = await requestAxios(`${MANAGER_IP}/check-status`);
        console.log(response)
        return response;
    }

);

export const getExperimentConfig = createAsyncThunk(
    'experiment/config', // 这个名字有什么意义呢？
    async () => {
        const response = await requestAxios(`${MANAGER_IP}/experiment`);
        console.log(response)
        return response;
    }

);

export const experimentSlice = createSlice({
    name: 'experiment',
    // `createSlice` will infer the state type from the `initialState` argument
    initialState,
    reducers: {
        increment: (state) => {
            state.status += 1
        },
        // Use the PayloadAction type to declare the contents of `action.payload`
        incrementByAmount: (state, action: PayloadAction<number>) => {
            state.status += action.payload
        },
    },
    extraReducers: (builder) => {
        builder
            .addCase(getExperimentstatus.pending, (state) => {
                console.log('异步请求正在进行中!')
            })
            .addCase(getExperimentstatus.rejected, (state) => {
                console.log('异步请求失败!')
            })
            .addCase(getExperimentstatus.fulfilled, (state, action) => {
                state.status = action.payload.status;
                if(action.payload.status === 'ERROR'){
                    state.error = action.payload.error[0]; // error 存放在error数组第一位
                }
            });
    }
});

export const { increment, incrementByAmount } = experimentSlice.actions

export default experimentSlice.reducer
