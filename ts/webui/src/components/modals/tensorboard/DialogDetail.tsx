import React from 'react';
import PropTypes from 'prop-types';
import { PrimaryButton, Dialog, DialogType, DialogFooter } from '@fluentui/react';

function DialogDetail(props): any {

    const {status, visible, func} = props;
    const dialogContentProps = {
        type: DialogType.normal,
        title: 'Tensorboard progress',
        closeButtonAriaLabel: 'Close',
        subText: `Please waiting some time to see tensorboard because this trial's tensorboard status is ${status}`,
    };
    console.info('dialog');

    return (
        <Dialog
            hidden={!visible}
            // onDismiss={toggleHideDialog}
            dialogContentProps={dialogContentProps}
            className='dialog'
        >
            <DialogFooter>
                <PrimaryButton onClick={() => {func(false)}} text="Close" />
            </DialogFooter>
        </Dialog>
    );
}

DialogDetail.propTypes = {
    status: PropTypes.string,
    visible: PropTypes.bool,
    func: PropTypes.func,
};

export default DialogDetail;
