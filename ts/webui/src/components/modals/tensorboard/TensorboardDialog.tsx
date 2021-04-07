import React from 'react';
import PropTypes from 'prop-types';
import { PrimaryButton, Dialog, DialogType, DialogFooter } from '@fluentui/react';

function TensorboardDialog(props): any {
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

    const startTensorboard = isReaptedStartTensorboard ? (
        <div>
            You had started this tensorBoard with these trials:
            <span className='bold'>{item.trialJobIdList.join(' ,')}</span>.
            <div className='line-height'>
                Its tensorBoard id: <span className='bold'>{item.id}</span>
            </div>
        </div>
    ) : (
        <div>
            You are starting a new TensorBoard with trials:
            <span className='bold'>{item.trialJobIdList.join(' ,')}</span>.
            <div className='line-height'>
                TensorBoard id: <span className='bold'>{item.id}</span>
            </div>
        </div>
    );

    return (
        <Dialog hidden={false} dialogContentProps={dialogContentProps} modalProps={{ className: 'dialog' }}>
            {errorMessage.error ? (
                <div>
                    <span>Failed to start tensorBoard! Error message: {errorMessage.message}</span>.
                </div>
            ) : isShowTensorboardDetail ? (
                <div>
                    This tensorBoard with trials: <span className='bold'>{item.trialJobIdList.join(', ')}</span>.
                </div>
            ) : (
                startTensorboard
            )}
            {errorMessage.error ? (
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

TensorboardDialog.propTypes = {
    isReaptedStartTensorboard: PropTypes.bool,
    isShowTensorboardDetail: PropTypes.bool,
    onHideDialog: PropTypes.func,
    item: PropTypes.object,
    errorMessage: PropTypes.object
};

export default TensorboardDialog;
