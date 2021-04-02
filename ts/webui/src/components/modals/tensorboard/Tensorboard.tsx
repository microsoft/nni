import React from 'react';
import PropTypes from 'prop-types';
import { PrimaryButton, Dialog, DialogType, DialogFooter } from '@fluentui/react';

function StartTensorboardDialog(props): any {

    const { isReaptedTensorboard, onHideDialog, item } = props;

    const dialogContentProps = {
        type: DialogType.normal,
        title: 'Tensorboard',
        closeButtonAriaLabel: 'OK',
    };

    function gotoTensorboard(): void {
        const hostname = window.location.hostname;
        const protocol = window.location.protocol;
        window.open(`${protocol}//${hostname}:${item.port}`);
        onHideDialog();
    }

    return (
        <Dialog
            hidden={false}
            dialogContentProps={dialogContentProps}
            className='dialog'
        >
            {
                isReaptedTensorboard
                ?
                <div>
                    You had started this tensorboard with these trials: <span className='bold'>{item.trialJobIdList.join(', ')}</span>.
                    <div className='line-height'>Its tensorboard id: <span className='bold'>{item.id}</span></div>
                </div>
    :
                <div>
                    You are starting a new Tensorboard with trials: <span className='bold'>{item.trialJobIdList.join(', ')}</span>.
                    <div className='line-height'>Tensorboard id: <span className='bold'>{item.id}</span></div>
                </div>
            }
            <DialogFooter>
                <PrimaryButton onClick={gotoTensorboard} text="OK" />
            </DialogFooter>
        </Dialog>
    );
}

StartTensorboardDialog.propTypes = {
    isReaptedTensorboard: PropTypes.bool,
    onHideDialog: PropTypes.func,
    item: PropTypes.object
};

export default StartTensorboardDialog;
