import React from 'react';
import PropTypes from 'prop-types';
import { PrimaryButton, Dialog, DialogType, DialogFooter } from '@fluentui/react';

function DialogDetail(props): any {

    const { visible, message, func } = props;
    const dialogContentProps = {
        type: DialogType.normal,
        title: 'Tensorboard',
        closeButtonAriaLabel: 'Close',
        subText: message
    };

    return (
        <Dialog
            hidden={!visible}
            // onDismiss={toggleHideDialog}
            dialogContentProps={dialogContentProps}
            className='dialog'
        >
            <DialogFooter>
                <PrimaryButton onClick={(): void => { func(false) }} text="Close" />
            </DialogFooter>
        </Dialog>
    );
}

DialogDetail.propTypes = {
    visible: PropTypes.bool,
    message: PropTypes.string,
    func: PropTypes.func,
};

export default DialogDetail;
