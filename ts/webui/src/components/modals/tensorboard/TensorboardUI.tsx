import React, {useState, useEffect} from 'react';
import PropTypes from 'prop-types';
import axios from 'axios';
import { DefaultButton, IContextualMenuProps } from '@fluentui/react';
import { MANAGER_IP } from '../../../static/const';
import { disableTensorboard, getTensorboardMenu } from '../../../static/function';
import { Tensorboard } from '../../../static/interface';
import StartTensorboardDialog from '../tensorboard/Tensorboard';
// import ShowTensorBoardDetail from '../tensorboard/ShowTensorBoardDetail';

function TensorboardUI(props): any {

    let refreshTensorboard = 0;
    const { selectedRowIds } = props;
    const [queryTensorboardList, setQueryTensorboardList] = useState([]);
    const [isReaptedTensorboard, setReaptedTensorboard] = useState(false);
    const [tensorboardPanelVisible, setTensorboardPanelVisible] = useState(false);
    const [isShowTensorboardDetail, setIsShowTensorboardDetail] = useState(false);
    const [selectedTensorboard, setSelectedTensorboard] = useState({});
    const [errorMessage, setErrorMessage] = useState({});
    const [timerList, setTimerList] = useState([0]);

    function startTrialTensorboard(): void {
        const { selectedRowIds } = props;
        setIsShowTensorboardDetail(false);
        const result = queryTensorboardList.filter((item: Tensorboard) => item.trialJobIdList.join(',') === selectedRowIds.join(','));
        if (result.length > 0) {
            setReaptedTensorboard(true);
            setSelectedTensorboard(result[0]);
            setTensorboardPanelVisible(true);
        } else {
            const startTensorboard = axios.post(`${MANAGER_IP}/tensorboard`, { trials: selectedRowIds.join(',') });
            startTensorboard.then(res => {
                if (res.status === 200) {
                    setSelectedTensorboard(res.data);
                    setTensorboardPanelVisible(true);
                    queryAllTensorboard();
                }
            }).catch(error => {
                setTensorboardPanelVisible(true);
                setErrorMessage({
                    error: true,
                    message: error.message || 'Tensorboard start failed'
                });
            });
            setReaptedTensorboard(false);
        }
    }

    function initQueryTensorboard(): void {
        const queryTensorboard = axios.get(`${MANAGER_IP}/tensorboard-tasks`);
        queryTensorboard.then(res => {
            if (res.status === 200) {
                console.info('***init***');
                setQueryTensorboardList(res.data);
            }
        });
    }

    function queryAllTensorboard (): void {
        // if(this.tableListComponent){
            const queryTensorboard = axios.get(`${MANAGER_IP}/tensorboard-tasks`);
            queryTensorboard.then(res => {
                if (res.status === 200) {
                    console.info('****************');
                    setQueryTensorboardList(res.data);
                    closeTimer();
                    refreshTensorboard = window.setTimeout(queryAllTensorboard, 10000);
                    const temp = timerList;
                    temp.push(refreshTensorboard);
                    setTimerList(temp);
                }
            }).catch(_error => {
                alert('Failed to start tensorboard');
            });
        // }
    }

    function closeTimer(): void {
        timerList.forEach(item => {
            window.clearTimeout(item);
        });
    }

    function stopAllTensorboard (): void {
        const delTensorboard = axios.delete(`${MANAGER_IP}/tensorboard-tasks`);
        delTensorboard.then(res => {
            if (res.status === 200) {
                setQueryTensorboardList([]);
                closeTimer();
            }
        });
    }

    function seeTensorboardWebportal (item: Tensorboard): void {
        setSelectedTensorboard(item);
        setIsShowTensorboardDetail(true);
        setTensorboardPanelVisible(true);
    }

    const isDisableTensorboardBtn = disableTensorboard(selectedRowIds, queryTensorboardList);
    const tensorboardMenu: IContextualMenuProps = getTensorboardMenu(queryTensorboardList, stopAllTensorboard, seeTensorboardWebportal);

    useEffect(() => {
        initQueryTensorboard();
    }, []);

    return (
        <React.Fragment>
            <DefaultButton
                text='TensorBoard'
                className='elementMarginLeft'
                split
                splitButtonAriaLabel="See 2 options"
                aria-roledescription="split button"
                menuProps={tensorboardMenu}
                onClick={(): void => startTrialTensorboard()}
                disabled={isDisableTensorboardBtn}
            />
            {
                queryTensorboardList.length !== 0 ?
                    <span className='circle'>{queryTensorboardList.length}</span>
                    : null
            }
            {tensorboardPanelVisible && (
                <StartTensorboardDialog
                    isReaptedTensorboard={isReaptedTensorboard}
                    isShowTensorboardDetail={isShowTensorboardDetail}
                    errorMessage={errorMessage}
                    item={selectedTensorboard}
                    onHideDialog={(): void => {
                        setTensorboardPanelVisible(false);
                    }}
                />
            )}
        </React.Fragment>
    );
}

TensorboardUI.propTypes = {
    selectedRowIds: PropTypes.array
};

export default TensorboardUI;
