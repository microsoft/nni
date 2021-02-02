import React from 'react';
import PropTypes from 'prop-types';
import { PrimaryButton, Dialog, DialogType, DialogFooter } from '@fluentui/react';

function DialogDetail(props): any {

    const { message, func } = props;
    const dialogContentProps = {
        type: DialogType.normal,
        title: 'Tensorboard',
        closeButtonAriaLabel: 'Close',
        subText: message
    };

    return (
        <Dialog
            hidden={false}
            // onDismiss={toggleHideDialog}
            dialogContentProps={dialogContentProps}
            className='dialog'
        >
            <DialogFooter>
                <PrimaryButton onClick={(): void => { func(false) || func() }} text="Close" />
            </DialogFooter>
        </Dialog>
    );
}

DialogDetail.propTypes = {
    message: PropTypes.string,
    func: PropTypes.func,
};

export default DialogDetail;
