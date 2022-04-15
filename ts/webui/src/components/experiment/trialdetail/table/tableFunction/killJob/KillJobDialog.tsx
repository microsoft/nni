import React from 'react';
import PropTypes from 'prop-types';
import { PrimaryButton, Dialog, DialogType, DialogFooter } from '@fluentui/react';

function KillJobDialog(props): any {
    const { onHideDialog, trialId, isError } = props;

    const dialogContentProps = {
        type: DialogType.normal,
        title: 'Kill job'
    };

    return (
        <Dialog hidden={false} dialogContentProps={dialogContentProps} modalProps={{ className: 'dialog' }}>
            {isError.isError ? (
                <div>
                    <div>Trial {trialId} kill failed!</div>
                    <span>Error message: {isError.message}</span>
                </div>
            ) : (
                <div>
                    Trial <span className='bold'>{trialId}</span> had been killed successfully.
                </div>
            )}
            <DialogFooter>
                <PrimaryButton onClick={onHideDialog} text='Close' />
            </DialogFooter>
        </Dialog>
    );
}

KillJobDialog.propTypes = {
    trialId: PropTypes.string,
    isError: PropTypes.object,
    onHideDialog: PropTypes.func
};

export default KillJobDialog;
