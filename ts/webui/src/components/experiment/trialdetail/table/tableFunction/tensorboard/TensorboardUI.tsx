import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { DefaultButton, IContextualMenuProps } from '@fluentui/react';
import { MANAGER_IP } from '@static/const';
import { disableTensorboard, getTensorboardMenu } from '@static/function';
import TensorboardDialog from './TensorboardDialog';
import { KillJobIsError, TensorboardTaskInfo } from '@static/interface';

interface TensorboardUIProps {
    selectedRowIds: string[];
    changeSelectTrialIds: () => void;
}

function TensorboardUI(props: TensorboardUIProps): any {
    let refreshTensorboard = 0;
    const { selectedRowIds, changeSelectTrialIds } = props;
    const [queryTensorboardList, setQueryTensorboardList] = useState([] as TensorboardTaskInfo[]);
    const [isReaptedStartTensorboard, setReaptedTensorboard] = useState(false);
    const [tensorboardPanelVisible, setTensorboardPanelVisible] = useState(false);
    const [isShowTensorboardDetail, setIsShowTensorboardDetail] = useState(false);
    const [selectedTensorboard, setSelectedTensorboard] = useState({} as TensorboardTaskInfo);
    const [errorMessage, setErrorMessage] = useState({} as KillJobIsError);
    const [timerList, setTimerList] = useState([0] as number[]);

    function startTrialTensorboard(): void {
        const { selectedRowIds } = props;
        if (selectedRowIds.length > 0) {
            setIsShowTensorboardDetail(false);
            const result = queryTensorboardList.filter(
                (item: TensorboardTaskInfo) => item.trialJobIdList.join(',') === selectedRowIds.join(',')
            );
            if (result.length > 0) {
                setReaptedTensorboard(true);
                setSelectedTensorboard(result[0]);
                setTensorboardPanelVisible(true);
            } else {
                const startTensorboard = axios.post(`${MANAGER_IP}/tensorboard`, { trials: selectedRowIds.join(',') });
                startTensorboard
                    .then(res => {
                        if (res.status === 200) {
                            setSelectedTensorboard(res.data);
                            closeTimer();
                            queryAllTensorboard();
                            setErrorMessage({ isError: false, message: '' });
                            setTensorboardPanelVisible(true);
                        }
                    })
                    .catch(err => {
                        if (err.response) {
                            setErrorMessage({
                                isError: true,
                                message: err.response.data.error || 'Failed to start tensorBoard!'
                            });
                        }
                        setTensorboardPanelVisible(true);
                    });
                setReaptedTensorboard(false);
            }
        } else {
            alert('Please select trials first!');
        }
    }

    function queryAllTensorboard(): void {
        const queryTensorboard = axios.get(`${MANAGER_IP}/tensorboard-tasks`);
        queryTensorboard.then(res => {
            if (res.status === 200) {
                setQueryTensorboardList(res.data);
                if (res.data.length !== 0) {
                    refreshTensorboard = window.setTimeout(queryAllTensorboard, 10000);
                    const storeTimerList = timerList;
                    storeTimerList.push(refreshTensorboard);
                    setTimerList(storeTimerList);
                }
            }
        });
    }

    function closeTimer(): void {
        timerList.forEach(item => {
            window.clearTimeout(item);
        });
    }

    function stopAllTensorboard(): void {
        const delTensorboard = axios.delete(`${MANAGER_IP}/tensorboard-tasks`);
        delTensorboard.then(res => {
            if (res.status === 200) {
                setQueryTensorboardList([]);
                closeTimer();
            }
        });
    }

    function seeTensorboardWebportal(item: TensorboardTaskInfo): void {
        setSelectedTensorboard(item);
        setIsShowTensorboardDetail(true);
        setTensorboardPanelVisible(true);
    }

    const isDisableTensorboardBtn = disableTensorboard(selectedRowIds, queryTensorboardList);
    const tensorboardMenu: IContextualMenuProps = getTensorboardMenu(
        queryTensorboardList,
        stopAllTensorboard,
        seeTensorboardWebportal
    );

    useEffect(() => {
        queryAllTensorboard();
        // clear timer when component is unmounted
        return function closeTimer(): void {
            timerList.forEach(item => {
                window.clearTimeout(item);
            });
        };
    }, []);

    return (
        <React.Fragment>
            <DefaultButton
                text='TensorBoard'
                split
                splitButtonAriaLabel='See 2 options'
                aria-roledescription='split button'
                menuProps={tensorboardMenu}
                onClick={(): void => startTrialTensorboard()}
                disabled={isDisableTensorboardBtn}
            />
            {queryTensorboardList.length !== 0 ? <span className='circle'>{queryTensorboardList.length}</span> : null}
            {tensorboardPanelVisible && (
                <TensorboardDialog
                    isReaptedStartTensorboard={isReaptedStartTensorboard}
                    isShowTensorboardDetail={isShowTensorboardDetail}
                    errorMessage={errorMessage}
                    item={selectedTensorboard}
                    onHideDialog={(): void => {
                        setTensorboardPanelVisible(false);
                        changeSelectTrialIds();
                    }}
                />
            )}
        </React.Fragment>
    );
}

export default TensorboardUI;
