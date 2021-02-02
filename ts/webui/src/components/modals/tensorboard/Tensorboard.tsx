import React, { useState, useEffect } from 'react';
import PropTypes from 'prop-types';
import { Stack, Panel, StackItem, PrimaryButton, DetailsList, IColumn, IconButton } from '@fluentui/react';
import DialogDetail from './DialogDetail';
import { caclMonacoEditorHeight, requestAxios } from '../../../static/function';
import { MANAGER_IP } from '../../../static/const';
import '../../../static/style/tensorboard.scss';

function Tensorboard(props): any {

    const { onHideDialog, trialIDs } = props;
    const [deleteIDs, setDeleteIDs] = useState([] as string[]);
    const [trialCount, setTrialCount] = useState(trialIDs.length - deleteIDs.length);
    const [source, setSource] = useState([]);
    const [status, setStatus] = useState(''); // trial tensorboard status
    const [visibleDialog, setVisibleDialog] = useState(false);

    const columns: IColumn[] = [
        {
            name: 'ID',
            key: 'id',
            fieldName: 'id',
            minWidth: 60,
            maxWidth: 120,
            isResizable: true,
            className: 'tableHead leftTitle',
            data: 'string',
            onRender: (item: any): React.ReactNode => <div className='succeed-padding'>{item.id}</div>
        },
        {
            name: 'Operation',
            key: '_operation',
            fieldName: 'operation',
            minWidth: 90,
            maxWidth: 90,
            isResizable: true,
            className: 'detail-table',
            onRender: _renderOperationColumn
        }
    ];

    useEffect(() => {
        const realIDs = trialIDs.filter(item => !(deleteIDs.includes(item)));
        setSource(realIDs.map(id => ({ id })));
    }, [trialCount]); // trialCount发生改变时触发页面更新

    // const { experiment, expDrawerHeight } = state;
    const tableHeight = caclMonacoEditorHeight(window.innerHeight);

    function _renderOperationColumn(record: any): React.ReactNode {
        return (
            <Stack className='operationBtn' horizontal>
                <IconButton
                    iconProps={{ iconName: 'OpenInNewWindow' }}
                    title="open"
                    onClick={() => openTrialTensorboard(record.id)}
                />
                <IconButton
                    iconProps={{ iconName: 'Delete' }}
                    title="delete"
                    onClick={() => deleteOneTrialTensorboard(record.id)}
                />
            </Stack>
        );
    }

    async function openTrialTensorboard(id: string): Promise<void> {
        /** Get tensorboard status
            Request: Get /api/v1/tensorboard/:id
            Response if success:
            Status:200
            {
                "status": "downloading data | running | stopping | stopped"
                "url": "tensorboard url"
            }
        */
    //    效果演示代码
    //    await setStatus('downloag');
    //    await setVisibleDialog(true);
       console.info(id);
        await requestAxios(`${MANAGER_IP}/tensorboard/:${id}`)
            .then(data => {
                if (data.status !== 'downloading data') {
                    // trial 启动成功
                    window.open(data.url);
                } else {
                    // 提示trial正在起tensorboard, 展示当前状态，
                    setStatus(data.status);
                    setVisibleDialog(true);
                }
            })
            .catch(_error => {
                // 页面展示error message
                alert('youwenti');
                // TODO: 提示有问题，请重新点击
            });
    }

    function deleteOneTrialTensorboard(id: string): void {
        /**
         * 	4. Stop tensorboard
                Request: DELETE /api/v1/tensorboard/:id
                Response if success
                {
                    status: "stopping"
                }
         */
        const a = deleteIDs;
        a.push(id);
        setDeleteIDs(a);
        setTrialCount(trialIDs.length - a.length);
        console.info('dele op');

    }

    /**
     * 	1. Start new tensorboard
            Request: POST /api/v1/tensorboard
            Parameters:
            {
                "trials": "trialId1, trialId2"
            }
            Response if success:
            Status:201
            {
            tensorboardId: "id"
            }
            Response if failed:
            Status 400
            {
            Message:"error message"
            }

     */
    return (
        <Panel isOpen={true} hasCloseButton={false} isLightDismiss={true} onLightDismissClick={onHideDialog}>
            <div className='panel'>
                <div className='panelName'>
                    <span>Tensorboard</span>
                    <span className='circle'>{trialCount}</span>
                </div>
                <DetailsList
                    columns={columns}
                    items={source}
                    setKey='set'
                    compact={true}
                    selectionMode={0}
                    className='succTable'
                    styles={{ root: { height: tableHeight } }}
                />
                <Stack horizontal className='buttons'>
                    <StackItem grow={12} className='close'>
                        <PrimaryButton text='Close' onClick={onHideDialog} />
                    </StackItem>
                </Stack>
            </div>
            {visibleDialog && <DialogDetail status={status} visible={visibleDialog} func={setVisibleDialog}/>}
        </Panel>
    );
}

Tensorboard.propTypes = {
    trialIDs: PropTypes.array,
    onHideDialog: PropTypes.func
};

export default Tensorboard;
