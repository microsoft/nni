import React from 'react';
// import React, { useState, useEffect } from 'react';
import PropTypes from 'prop-types';
// import { MANAGER_IP } from '../../../static/const';
import { PrimaryButton, Dialog, DialogType, DialogFooter } from '@fluentui/react';
// import { caclMonacoEditorHeight, requestAxios } from '../../../static/function';

function ShowTensorBoardDetail(props): any {

    const { onHideDialog, item } = props;

    const dialogContentProps = {
        type: DialogType.normal,
        title: item.id,
        closeButtonAriaLabel: 'Close',
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
            <div>
                This tensorboard with trials: <span className='bold'>{item.trialJobIdList.join(', ')}</span>.
            </div>
            <DialogFooter>
                <PrimaryButton onClick={gotoTensorboard} text="See tensorboard" />
            </DialogFooter>
        </Dialog>
    );
}

ShowTensorBoardDetail.propTypes = {
    item: PropTypes.object,
    onHideDialog: PropTypes.func
};

export default ShowTensorBoardDetail;
