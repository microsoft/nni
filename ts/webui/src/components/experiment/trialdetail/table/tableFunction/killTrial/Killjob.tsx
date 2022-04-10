import * as React from 'react';
import { Stack, FocusTrapCallout, DefaultButton, FocusZone, PrimaryButton } from '@fluentui/react';
import { killJob } from '@static/function';
import { blocked } from '@components/fluent/Icon';
import { styles } from '@components/experiment/overview/params/basicInfoStyles';

interface KillJobState {
    isCalloutVisible: boolean;
}

interface KillJobProps {
    trial: any;
}

class KillJob extends React.Component<KillJobProps, KillJobState> {
    private menuButtonElement!: HTMLElement | null;
    constructor(props: KillJobProps) {
        super(props);
        this.state = { isCalloutVisible: false };
    }

    render(): React.ReactNode {
        const { isCalloutVisible } = this.state;
        const prompString = 'Are you sure to cancel this trial?';
        return (
            <div>
                <div className={styles.buttonArea} ref={(menuButton): any => (this.menuButtonElement = menuButton)}>
                    <PrimaryButton className='detail-button-operation' onClick={this.openPromot} title='kill'>
                        {blocked}
                    </PrimaryButton>
                </div>
                {isCalloutVisible ? (
                    <div>
                        <FocusTrapCallout
                            role='alertdialog'
                            className={styles.callout}
                            gapSpace={0}
                            target={this.menuButtonElement}
                            onDismiss={this.onDismiss}
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
                                <Stack className={styles.buttons} gap={8} horizontal>
                                    <DefaultButton onClick={this.onDismiss}>No</DefaultButton>
                                    <PrimaryButton onClick={this.onKill}>Yes</PrimaryButton>
                                </Stack>
                            </FocusZone>
                        </FocusTrapCallout>
                    </div>
                ) : null}
            </div>
        );
    }

    private onDismiss = (): void => {
        this.setState(() => ({ isCalloutVisible: false }));
    };

    private onKill = (): void => {
        this.setState({ isCalloutVisible: false }, () => {
            const { trial } = this.props;
            killJob(trial.key, trial.id, trial.status);
        });
    };

    private openPromot = (event: React.SyntheticEvent<EventTarget>): void => {
        event.preventDefault();
        event.stopPropagation();
        this.setState({ isCalloutVisible: true });
    };
}

export default KillJob;
