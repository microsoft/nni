import React, { useState, useContext, useCallback } from 'react';
import { Stack, Callout, Link, IconButton, IStackTokens } from '@fluentui/react';
import { useId } from '@uifabric/react-hooks';
import { EXPERIMENT } from '@static/datamodel';
import { getPrefix } from '@static/function';
import LogPanel from '@components/nav/slideNav/LogPanel';
import { BestMetricContext } from '../Overview';
import { styles } from '../../../common/calloutStyles';
import Config from './Config';
import '@style/common/experimentStatusColor.scss';
import '@style/experiment/overview/basic.scss';

const focusMetricGap: IStackTokens = {
    childrenGap: 4
};

export const BasicInfo = (): any => {
    const labelId: string = useId('callout-label');
    const descriptionId: string = useId('callout-description');
    const ref = React.createRef<HTMLDivElement>();
    const [isCalloutVisible, setCalloutVisible] = useState(false);
    const [isShowLogPanel, setShowLogPanel] = useState(false);
    const onDismiss = useCallback(() => setCalloutVisible(false), []);
    const showCallout = useCallback(() => setCalloutVisible(true), []);

    const closeLogPanel = useCallback(() => setShowLogPanel(false), []);
    const ShowLogPanel = useCallback(() => setShowLogPanel(true), []);
    const { bestAccuracy } = useContext(BestMetricContext);
    return (
        <div>
            <Stack horizontal className='experimentHead'>
                <div className='icon'>
                    <img src={(getPrefix() || '') + '/icons/experiment-icon.png'} />
                </div>
                <div>
                    <h3 className='title'>{EXPERIMENT.profile.params.experimentName || 'Experiment name'}</h3>
                    <div className='id'>{EXPERIMENT.profile.id}</div>
                </div>
            </Stack>
            <Stack horizontal horizontalAlign='space-between' tokens={focusMetricGap} className='focus-status-metric'>
                <div
                    style={{
                        backgroundImage: `url(${(getPrefix() || '') + '/icons/experiment-status.png'})`,
                        backgroundRepeat: 'no-repeat'
                    }}
                >
                    <span className={`${EXPERIMENT.status} status-text focus-text`}>{EXPERIMENT.status}</span>
                    {EXPERIMENT.status === 'ERROR' ? (
                        <div>
                            <div className={`${styles.buttonArea} error-info-icon`} ref={ref}>
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
                                    <div className={`${styles.header} font`}>
                                        <p className={`${styles.title} color333`} id={labelId}>
                                            Error
                                        </p>
                                    </div>
                                    <div className={`${styles.inner} font`}>
                                        <p className={`${styles.subtext} color333`} id={descriptionId}>
                                            {EXPERIMENT.error}
                                        </p>
                                        <div className={styles.actions}>
                                            <Link className={styles.link} onClick={ShowLogPanel}>
                                                Learn more
                                            </Link>
                                        </div>
                                    </div>
                                </Callout>
                            )}
                        </div>
                    ) : null}
                    <div className='description'>Status</div>
                </div>
                <div
                    style={{
                        backgroundImage: `url(${(getPrefix() || '') + '/icons/best-metric-bg.png'})`,
                        backgroundRepeat: 'no-repeat'
                    }}
                >
                    <div className='bestMetric'>
                        <div className={`${EXPERIMENT.status} focus-text`}>
                            {isNaN(bestAccuracy) ? 'N/A' : bestAccuracy.toFixed(6)}
                        </div>
                        <div className='description'>Best metric</div>
                    </div>
                </div>
            </Stack>
            {/* learn about click -> default active key is dispatcher. */}
            {isShowLogPanel ? <LogPanel closePanel={closeLogPanel} activeTab='dispatcher' /> : null}
            <Config />
        </div>
    );
};
