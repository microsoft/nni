import React, {useState} from 'react';
import PropTypes from 'prop-types';
import axios from 'axios';
import { DefaultButton, IContextualMenuProps } from '@fluentui/react';
import { MANAGER_IP } from '../../../static/const';
import { disableTensorboard, getTensorboardMenu } from '../../../static/function';
import { Tensorboard } from '../../../static/interface';
import StartTensorboardDialog from '../tensorboard/Tensorboard';
import ShowTensorBoardDetail from '../tensorboard/ShowTensorBoardDetail';

function TensorboardUI(props): any {

    let refreshTensorboard = 0;
    const { selectedRowIds } = props;
    const [queryTensorboardList, setQueryTensorboardList] = useState([]);
    const [isReaptedTensorboard, setReaptedTensorboard] = useState(false);
    const [tensorboardPanelVisible, setTensorboardPanelVisible] = useState(false);
    const [visibleDialog, setVisibleDialog] = useState(false);
    const [detailTensorboardPanelVisible, setDetailTensorboardPanelVisible] = useState(false);
    const [selectedTensorboard, setSelectedTensorboard] = useState({});
    const [dialogContent, setDialogContent] = useState('');
    const [timerList, setTimerList] = useState([0]);

    function startTrialTensorboard(): void {
        const { selectedRowIds } = props;
        
        const result = queryTensorboardList.filter((item: Tensorboard) => item.trialJobIdList.join(',') === selectedRowIds.join(','));
        if (result.length > 0) {
            setReaptedTensorboard(true);
            setSelectedTensorboard(result[0]);
            setTensorboardPanelVisible(true);
        } else {
            const startTensorboard = axios.post(`${MANAGER_IP}/tensorboard`, { trials: selectedRowIds.join(',') });
            startTensorboard.then(res => {
                if (res.status === 200) {
                    // setReaptedTensorboard(false);
                    setSelectedTensorboard(res.data);
                    setTensorboardPanelVisible(true);
                    queryAllTensorboard();
                }
            }).catch(error => {
                setVisibleDialog(true);
                setDialogContent(error.message || 'Tensorboard start failed');
            });
            setReaptedTensorboard(false);
        }
    }

    function queryAllTensorboard (): void {
        // if(this.tableListComponent){
            const queryTensorboard = axios.get(`${MANAGER_IP}/tensorboard-tasks`);
            queryTensorboard.then(res => {
                if (res.status === 200) {
                    console.info('****************');
                    setQueryTensorboardList(res.data);
                    refreshTensorboard = window.setTimeout(queryAllTensorboard, 10000);
                    setTimerList(timerList.push(refreshTensorboard) as any);
                    console.info('list', timerList);
                }
            }).catch(_error => {
                alert('Failed to start tensorboard');
            });
        // }
    }

    function stopAllTensorboard (): void {
        const delTensorboard = axios.delete(`${MANAGER_IP}/tensorboard-tasks`);
        delTensorboard.then(res => {
            if (res.status === 200) {
                setQueryTensorboardList([]);
                console.info('stop list', timerList);
                timerList.forEach(item => {
                    window.clearTimeout(item);
                });
                console.info('--------------');
            }
        });
    }

    function seeTensorboardWebportal (item: Tensorboard): void {
        setSelectedTensorboard(item);
        setDetailTensorboardPanelVisible(true);
    }

    const isDisableTensorboardBtn = disableTensorboard(selectedRowIds, queryTensorboardList);
    const tensorboardMenu: IContextualMenuProps = getTensorboardMenu(queryTensorboardList, stopAllTensorboard, seeTensorboardWebportal);
    
    // useEffect(() => {
    //     timerList.forEach(item => {
    //         console.info('来请定时器');
    //         console.info(item);
    //         window.clearTimeout(item);
    //     });
    // }, [closeTimer]);

    console.info(visibleDialog);
    console.info(dialogContent);
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
                        item={selectedTensorboard}
                        onHideDialog={(): void => {
                            setTensorboardPanelVisible(false);
                        }}
                    />
                )}
                {detailTensorboardPanelVisible && (
                    <ShowTensorBoardDetail
                        item={selectedTensorboard}
                        onHideDialog={(): void => {
                            setDetailTensorboardPanelVisible(false);
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
