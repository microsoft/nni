import React from 'react';
import { PrimaryButton, Dialog, DialogType, DialogFooter } from '@fluentui/react';
import { KillJobIsError, TensorboardTaskInfo } from '@static/interface';

interface TensorboardDialogProps {
    isReaptedStartTensorboard: boolean;
    isShowTensorboardDetail: boolean;
    onHideDialog: () => void;
    item: TensorboardTaskInfo;
    errorMessage: KillJobIsError;
}

function TensorboardDialog(props: TensorboardDialogProps): any {
    const { isReaptedStartTensorboard, onHideDialog, item, isShowTensorboardDetail, errorMessage } = props;

    const dialogContentProps = {
        type: DialogType.normal,
        title: `${isShowTensorboardDetail ? item.id : 'TensorBoard'}`
    };

    function gotoTensorboard(): void {
        const hostname = window.location.hostname;
        const protocol = window.location.protocol;
        window.open(`${protocol}//${hostname}:${item.port}`);
        onHideDialog();
    }

    return (
        <Dialog hidden={false} dialogContentProps={dialogContentProps} modalProps={{ className: 'dialog' }}>
            {errorMessage.isError ? (
                <div>
                    <span>Error message: {errorMessage.message}</span>
                </div>
            ) : isShowTensorboardDetail ? (
                <div>
                    This tensorBoard with trials: <span className='bold'>{item.trialJobIdList.join(', ')}</span>.
                </div>
            ) : isReaptedStartTensorboard ? (
                <div>
                    You had started this tensorBoard with these trials:
                    <span className='bold'>{item.trialJobIdList.join(', ')}</span>.
                    <div className='line-height'>
                        Its tensorBoard id: <span className='bold'>{item.id}</span>
                    </div>
                </div>
            ) : (
                <div>
                    You are starting a new TensorBoard with trials:
                    <span className='bold'>{item.trialJobIdList.join(', ')}</span>.
                    <div className='line-height'>
                        TensorBoard id: <span className='bold'>{item.id}</span>
                    </div>
                </div>
            )}
            {errorMessage.isError ? (
                <DialogFooter>
                    <PrimaryButton onClick={onHideDialog} text='Close' />
                </DialogFooter>
            ) : (
                <DialogFooter>
                    <PrimaryButton
                        onClick={gotoTensorboard}
                        text={`${isShowTensorboardDetail ? 'See tensorBoard' : 'Ok'}`}
                    />
                </DialogFooter>
            )}
        </Dialog>
    );
}

export default TensorboardDialog;
