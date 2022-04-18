import React, { useState, useRef, useContext } from 'react';
import PropTypes from 'prop-types';
import axios from 'axios';
import { Stack, FocusTrapCallout, DefaultButton, FocusZone, PrimaryButton } from '@fluentui/react';
import { MANAGER_IP } from '@static/const';
import KillJobDialog from './KillJobDialog';
import { blocked } from '@components/fluent/Icon';
import { gap10 } from '@components/fluent/ChildrenGap';
import { styles } from '@components/experiment/overview/params/basicInfoStyles';
import { AppContext } from '@/App';

function KillJobIndex(props): any {
    const menuButtonElement = useRef(null);
    const { startTimer, closeTimer, refreshDetailTable } = useContext(AppContext);
    const { trialId } = props;
    const [isCalloutVisible, setCalloutVisible] = useState(false);
    const [isVisibleKillDialog, setKillDialogVisible] = useState(false);
    const [error, setError] = useState({ isError: false, message: '' });

    const prompString = 'Are you sure to cancel this trial?';

    // kill trial
    const killJob = (id: string): void => {
        closeTimer(); // close auto refresh to confirm show the kill model
        axios(`${MANAGER_IP}/trial-jobs/${id}`, {
            method: 'DELETE',
            headers: {
                'Content-Type': 'application/json;charset=utf-8'
            }
        })
            .then(res => {
                if (res.status === 200) {
                    setKillDialogVisible(true);
                    setError({ isError: false, message: '' });
                } else {
                    setKillDialogVisible(true);
                    setError({ isError: false, message: 'fail to cancel the job' });
                }
            })
            .catch(error => {
                if (error.response) {
                    setKillDialogVisible(true);
                    setError({ isError: false, message: error.response.data.error || 'Fail to cancel the job' });
                } else {
                    setKillDialogVisible(true);
                    setError({
                        isError: false,
                        message: error.response.data.error || '500 error, fail to cancel the job'
                    });
                }
            });
    };

    const onDismissKillJobMessageDialog = async (): Promise<void> => {
        setKillDialogVisible(false);
        await refreshDetailTable();
        startTimer(); // start refresh
    };

    const onDismiss = (): void => {
        setCalloutVisible(false);
    };

    const onKill = (): void => {
        setCalloutVisible(false);
        killJob(props.trialId);
    };

    const openPromot = (event: React.SyntheticEvent<EventTarget>): void => {
        event.preventDefault();
        event.stopPropagation();
        setCalloutVisible(true);
    };

    return (
        <div>
            <div className={styles.buttonArea} ref={menuButtonElement}>
                <PrimaryButton className='detail-button-operation' onClick={openPromot} title='kill'>
                    {blocked}
                </PrimaryButton>
            </div>
            {isCalloutVisible ? (
                <div>
                    <FocusTrapCallout
                        role='alertdialog'
                        className={styles.callout}
                        gapSpace={0}
                        target={menuButtonElement}
                        onDismiss={onDismiss}
                        setInitialFocus={true}
                    >
                        <div className={styles.header}>
                            <p className={styles.title} style={{ color: '#333' }}>
                                Kill trial
                            </p>
                        </div>
                        <div className={styles.inner}>
                            <div>
                                <p className={styles.subtext} style={{ color: '#333' }}>
                                    {prompString}
                                </p>
                            </div>
                        </div>
                        <FocusZone>
                            <Stack className={styles.buttons} tokens={gap10} horizontal>
                                <DefaultButton onClick={onDismiss}>No</DefaultButton>
                                <PrimaryButton onClick={onKill}>Yes</PrimaryButton>
                            </Stack>
                        </FocusZone>
                    </FocusTrapCallout>
                </div>
            ) : null}
            {/* kill job status dialog */}
            {isVisibleKillDialog && (
                <KillJobDialog trialId={trialId} isError={error} onHideDialog={onDismissKillJobMessageDialog} />
            )}
        </div>
    );
}

KillJobIndex.propTypes = {
    trialId: PropTypes.string,
    updatePage: PropTypes.func
};

export default KillJobIndex;
