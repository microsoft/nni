import * as React from 'react';
import {
    Stack,
    FocusTrapCallout,
    DefaultButton,
    FocusZone,
    PrimaryButton,
    getTheme,
    mergeStyleSets,
    FontWeights
} from '@fluentui/react';
import { killJob } from '../../static/function';
import { blocked } from '../buttons/Icon';

const theme = getTheme();
const styles = mergeStyleSets({
    buttonArea: {
        verticalAlign: 'top',
        display: 'inline-block',
        textAlign: 'center',
        height: 32
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
    buttons: {
        display: 'flex',
        justifyContent: 'flex-end',
        padding: '0 24px 24px'
    },
    subtext: [
        theme.fonts.small,
        {
            margin: 0,
            color: theme.palette.neutralPrimary,
            fontWeight: FontWeights.semilight
        }
    ]
});

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

    onDismiss = (): void => {
        this.setState(() => ({ isCalloutVisible: false }));
    };

    onKill = (): void => {
        this.setState({ isCalloutVisible: false }, () => {
            const { trial } = this.props;
            killJob(trial.key, trial.id, trial.status);
        });
    };

    openPromot = (event: React.SyntheticEvent<EventTarget>): void => {
        event.preventDefault();
        event.stopPropagation();
        this.setState({ isCalloutVisible: true });
    };

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
}

export default KillJob;
