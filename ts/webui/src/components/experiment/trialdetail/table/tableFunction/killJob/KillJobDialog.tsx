import React from 'react';
import { PrimaryButton, Dialog, DialogType, DialogFooter } from '@fluentui/react';
import { KillJobIsError } from '@static/interface';

interface KillJobDialogProps {
    trialId: string;
    isError: KillJobIsError;
    onHideDialog: () => void;
}

function KillJobDialog(props: KillJobDialogProps): any {
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

export default KillJobDialog;
