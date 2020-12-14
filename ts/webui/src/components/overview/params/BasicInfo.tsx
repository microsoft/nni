import React, { useState, useCallback } from 'react';
import { Stack, Callout, Link, IconButton } from '@fluentui/react';
import LogDrawer from '../../modals/LogPanel';
import { EXPERIMENT } from '../../../static/datamodel';
import { formatTimestamp } from '../../../static/function';
import { useId } from '@uifabric/react-hooks';
import { BestMetricContext } from '../../Overview';
import { styles } from './basicInfoStyles';
import '../../../static/style/overview/probar.scss';
import '../../../static/style/overview/basic.scss';

export const BasicInfo = (): any => {
    const labelId: string = useId('callout-label');
    const descriptionId: string = useId('callout-description');
    const ref = React.createRef<HTMLDivElement>();
    const [isCalloutVisible, setCalloutVisible] = useState(false);
    const [isShowLogDrawer, setShowLogDrawer] = useState(false);
    const onDismiss = useCallback(() => setCalloutVisible(false), []);
    const showCallout = useCallback(() => setCalloutVisible(true), []);

    const closeLogDrawer = useCallback(() => setShowLogDrawer(false), []);
    const ShowLogDrawer = useCallback(() => setShowLogDrawer(true), []);

    return (
        <div>
            <Stack horizontal horizontalAlign='space-between' className='marginTop'>
                <div className='basic'>
                    <p>Name</p>
                    <div className='ellipsis'>{EXPERIMENT.profile.params.experimentName}</div>
                    <p className='marginTop'>ID</p>
                    <div className='ellipsis'>{EXPERIMENT.profile.id}</div>
                </div>
                <div className='basic'>
                    <p>Status</p>
                    <Stack horizontal className='status'>
                        <span className={`${EXPERIMENT.status} status-text`}>{EXPERIMENT.status}</span>
                        {EXPERIMENT.status === 'ERROR' ? (
                            <div>
                                <div className={styles.buttonArea} ref={ref}>
                                    <IconButton
                                        iconProps={{ iconName: 'info' }}
                                        onClick={isCalloutVisible ? onDismiss : showCallout}
                                    />
                                </div>
                                {isCalloutVisible && (
                                    <Callout
                                        className={styles.callout}
                                        ariaLabelledBy={labelId}
                                        ariaDescribedBy={descriptionId}
                                        role='alertdialog'
                                        gapSpace={0}
                                        target={ref}
                                        onDismiss={onDismiss}
                                        setInitialFocus={true}
                                    >
                                        <div className={styles.header}>
                                            <p className={`${styles.title} color`} id={labelId}>
                                                Error
                                            </p>
                                        </div>
                                        <div className={styles.inner}>
                                            <p className={`${styles.subtext} color`} id={descriptionId}>
                                                {EXPERIMENT.error}
                                            </p>
                                            <div className={styles.actions}>
                                                <Link className={styles.link} onClick={ShowLogDrawer}>
                                                    Learn about
                                                </Link>
                                            </div>
                                        </div>
                                    </Callout>
                                )}
                            </div>
                        ) : null}
                    </Stack>
                    <BestMetricContext.Consumer>
                        {(value): React.ReactNode => (
                            <Stack className='bestMetric'>
                                <p className='marginTop'>Best metric</p>
                                <div className={EXPERIMENT.status}>
                                    {isNaN(value.bestAccuracy) ? 'N/A' : value.bestAccuracy.toFixed(6)}
                                </div>
                            </Stack>
                        )}
                    </BestMetricContext.Consumer>
                </div>
                <div className='basic'>
                    <p>Start time</p>
                    <div className='ellipsis'>{formatTimestamp(EXPERIMENT.profile.startTime)}</div>
                    <p className='marginTop'>End time</p>
                    <div className='ellipsis'>{formatTimestamp(EXPERIMENT.profile.endTime)}</div>
                </div>
            </Stack>
            {/* learn about click -> default active key is dispatcher. */}
            {isShowLogDrawer ? <LogDrawer closeDrawer={closeLogDrawer} activeTab='dispatcher' /> : null}
        </div>
    );
};
