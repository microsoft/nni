import * as React from 'react';
import { Stack, Callout, Link, IconButton, FontWeights, mergeStyleSets, getId, getTheme, StackItem } from 'office-ui-fabric-react';
import axios from 'axios';
import { MANAGER_IP } from '../../static/const';
import { EXPERIMENT, TRIALS } from '../../static/datamodel';
import { convertTime } from '../../static/function';
import ConcurrencyInput from './NumInput';
import ProgressBar from './ProgressItem';
import LogDrawer from '../Modals/LogDrawer';
import MessageInfo from '../Modals/MessageInfo';
import '../../static/style/progress.scss';
import '../../static/style/probar.scss';
interface ProgressProps {
    concurrency: number;
    bestAccuracy: number;
    changeConcurrency: (val: number) => void;
    experimentUpdateBroadcast: number;
}

interface ProgressState {
    isShowLogDrawer: boolean;
    isCalloutVisible?: boolean;
    isShowSucceedInfo: boolean;
    info: string;
    typeInfo: string;
}

const itemStyles: React.CSSProperties = {
    height: 50,
    width: 100
};
const theme = getTheme();
const styles = mergeStyleSets({
    buttonArea: {
        verticalAlign: 'top',
        display: 'inline-block',
        textAlign: 'center',
        // margin: '0 100px',
        minWidth: 30,
        height: 30
    },
    callout: {
        maxWidth: 300
    },
    header: {
        padding: '18px 24px 12px'
    },
    title: [
        theme.fonts.xLarge,
        {
            margin: 0,
            color: theme.palette.neutralPrimary,
            fontWeight: FontWeights.semilight
        }
    ],
    inner: {
        height: '100%',
        padding: '0 24px 20px'
    },
    actions: {
        position: 'relative',
        marginTop: 20,
        width: '100%',
        whiteSpace: 'nowrap'
    },
    subtext: [
        theme.fonts.small,
        {
            margin: 0,
            color: theme.palette.neutralPrimary,
            fontWeight: FontWeights.semilight
        }
    ],
    link: [
        theme.fonts.medium,
        {
            color: theme.palette.neutralPrimary
        }
    ]
});

class Progressed extends React.Component<ProgressProps, ProgressState> {
    private _menuButtonElement!: HTMLDivElement | null;
    private _labelId: string = getId('callout-label');
    private _descriptionId: string = getId('callout-description');
    constructor(props: ProgressProps) {
        super(props);
        this.state = {
            isShowLogDrawer: false,
            isCalloutVisible: false,
            isShowSucceedInfo: false,
            info: '',
            typeInfo: 'success'
        };
    }

    hideSucceedInfo = (): void => {
        this.setState(() => ({ isShowSucceedInfo: false }));
    }
    
    /**
     * info: message content
     * typeInfo: message type: success | error...
     * continuousTime: show time, 2000ms 
     */
    getMessageInfo = (info: string, typeInfo: string): void => {
        this.setState(() => ({
            info, typeInfo,
            isShowSucceedInfo: true
        }));
        setTimeout(this.hideSucceedInfo, 2000);
    }

    editTrialConcurrency = async (userInput: string): Promise<void> => {
        if (!userInput.match(/^[1-9]\d*$/)) {
            this.getMessageInfo('Please enter a positive integer!', 'error');
            return;
        }
        const newConcurrency = parseInt(userInput, 10);
        if (newConcurrency === this.props.concurrency) {
            this.getMessageInfo('Trial concurrency has not changed', 'error');
            return;
        }

        const newProfile = Object.assign({}, EXPERIMENT.profile);
        newProfile.params.trialConcurrency = newConcurrency;

        // rest api, modify trial concurrency value
        try {
            const res = await axios.put(`${MANAGER_IP}/experiment`, newProfile, {
                // eslint-disable-next-line @typescript-eslint/camelcase
                params: { update_type: 'TRIAL_CONCURRENCY' }
            });
            if (res.status === 200) {
                this.getMessageInfo('Successfully updated trial concurrency', 'success');
                // NOTE: should we do this earlier in favor of poor networks?
                this.props.changeConcurrency(newConcurrency);
            }
        } catch (error) {
            if (error.response && error.response.data.error) {
                this.getMessageInfo(`Failed to update trial concurrency\n${error.response.data.error}`, 'error');
            } else if (error.response) {
                this.getMessageInfo(`Failed to update trial concurrency\nServer responsed ${error.response.status}`, 'error');
            } else if (error.message) {
                this.getMessageInfo(`Failed to update trial concurrency\n${error.message}`, 'error');
            } else {
                this.getMessageInfo(`Failed to update trial concurrency\nUnknown error`, 'error');
            }
        }
    }

    isShowDrawer = (): void => {
        this.setState({ isShowLogDrawer: true });
    }

    closeDrawer = (): void => {
        this.setState({ isShowLogDrawer: false });
    }

    _onDismiss = (): void => {
        this.setState({ isCalloutVisible: false });
    }

    _onShow = (): void => {
        this.setState({ isCalloutVisible: true });
    }

    render(): React.ReactNode {
        const { bestAccuracy } = this.props;
        const { isShowLogDrawer, isCalloutVisible, isShowSucceedInfo, info, typeInfo } = this.state;

        const count = TRIALS.countStatus();
        // eslint-disable-next-line @typescript-eslint/no-non-null-assertion
        const stoppedCount = count.get('USER_CANCELED')! + count.get('SYS_CANCELED')! + count.get('EARLY_STOPPED')!;
        // eslint-disable-next-line @typescript-eslint/no-non-null-assertion
        const bar2 = count.get('RUNNING')! + count.get('SUCCEEDED')! + count.get('FAILED')! + stoppedCount;
        // support type [0, 1], not 98%
        const bar2Percent = bar2 / EXPERIMENT.profile.params.maxTrialNum;
        const percent = EXPERIMENT.profile.execDuration / EXPERIMENT.profile.params.maxExecDuration;
        const remaining = convertTime(EXPERIMENT.profile.params.maxExecDuration - EXPERIMENT.profile.execDuration);
        const maxDuration = convertTime(EXPERIMENT.profile.params.maxExecDuration);
        const maxTrialNum = EXPERIMENT.profile.params.maxTrialNum;
        const execDuration = convertTime(EXPERIMENT.profile.execDuration);

        return (
            <Stack className="progress" id="barBack">
                <Stack className="basic lineBasic">
                    <p>Status</p>
                    <Stack horizontal className="status">
                        <span className={`${EXPERIMENT.status} status-text`}>{EXPERIMENT.status}</span>
                        {
                            EXPERIMENT.status === 'ERROR'
                                ?
                                <div>
                                    <div className={styles.buttonArea} ref={(val): any => this._menuButtonElement = val}>
                                        <IconButton
                                            iconProps={{ iconName: 'info' }}
                                            onClick={isCalloutVisible ? this._onDismiss : this._onShow}
                                        />
                                    </div>
                                    {isCalloutVisible && (
                                        <Callout
                                            className={styles.callout}
                                            ariaLabelledBy={this._labelId}
                                            ariaDescribedBy={this._descriptionId}
                                            role="alertdialog"
                                            gapSpace={0}
                                            target={this._menuButtonElement}
                                            onDismiss={this._onDismiss}
                                            setInitialFocus={true}
                                        >
                                            <div className={styles.header}>
                                                <p className={styles.title} id={this._labelId}>Error</p>
                                            </div>
                                            <div className={styles.inner}>
                                                <p className={styles.subtext} id={this._descriptionId}>
                                                    {EXPERIMENT.error}
                                                </p>
                                                <div className={styles.actions}>
                                                    <Link className={styles.link} onClick={this.isShowDrawer}>
                                                        Learn about
                                                    </Link>
                                                </div>
                                            </div>
                                        </Callout>
                                    )}
                                </div>
                                :
                                null
                        }
                    </Stack>
                </Stack>
                <ProgressBar
                    who="Duration"
                    percent={percent}
                    description={execDuration}
                    bgclass={EXPERIMENT.status}
                    maxString={`Max duration: ${maxDuration}`}
                />
                <ProgressBar
                    who="Trial numbers"
                    percent={bar2Percent}
                    description={bar2.toString()}
                    bgclass={EXPERIMENT.status}
                    maxString={`Max trial number: ${maxTrialNum}`}
                />
                <Stack className="basic colorOfbasic mess" horizontal>
                    <StackItem grow={50}>
                        <p>Best metric</p>
                        <div>{isNaN(bestAccuracy) ? 'N/A' : bestAccuracy.toFixed(6)}</div>
                    </StackItem>
                    <StackItem>
                        {isShowSucceedInfo && <MessageInfo className="info" typeInfo={typeInfo} info={info} />}
                    </StackItem>
                </Stack>
                <Stack horizontal horizontalAlign="space-between" className="mess">
                    <span style={itemStyles} className="basic colorOfbasic">
                        <p>Spent</p>
                        <div>{execDuration}</div>
                    </span>
                    <span style={itemStyles} className="basic colorOfbasic">
                        <p>Remaining</p>
                        <div className="time">{remaining}</div>
                    </span>
                    <span style={itemStyles}>
                        {/* modify concurrency */}
                        <p>Concurrency</p>
                        <ConcurrencyInput value={this.props.concurrency} updateValue={this.editTrialConcurrency} />
                    </span>
                    <span style={itemStyles} className="basic colorOfbasic"></span>
                </Stack>
                <Stack horizontal horizontalAlign="space-between" className="mess">
                    <div style={itemStyles} className="basic colorOfbasic">
                        <p>Running</p>
                        <div>{count.get('RUNNING')}</div>
                    </div>
                    <div style={itemStyles} className="basic colorOfbasic">
                        <p>Succeeded</p>
                        <div>{count.get('SUCCEEDED')}</div>
                    </div>
                    <div style={itemStyles} className="basic">
                        <p>Stopped</p>
                        <div>{stoppedCount}</div>
                    </div>
                    <div style={itemStyles} className="basic">
                        <p>Failed</p>
                        <div>{count.get('FAILED')}</div>
                    </div>
                </Stack>
                {/* learn about click -> default active key is dispatcher. */}
                {isShowLogDrawer ? (
                    <LogDrawer
                        closeDrawer={this.closeDrawer}
                        activeTab="dispatcher"
                    />
                ) : null}
            </Stack>
        );
    }

}

export default Progressed;
