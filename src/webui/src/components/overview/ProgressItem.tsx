import * as React from 'react';
import { Stack, StackItem, ProgressIndicator } from '@fluentui/react';

interface ProItemProps {
    who: string;
    percent: number;
    description: string;
    maxString: string;
    bgclass: string;
}

class ProgressBar extends React.Component<ProItemProps, {}> {

    constructor(props: ProItemProps) {
        super(props);

    }

    render(): React.ReactNode {
        const { who, percent, description, maxString, bgclass } = this.props;
        return (
            <div>
                <Stack horizontal className={`probar ${bgclass}`}>
                    <div className="name">{who}</div>
                    <div className="showProgress" style={{ width: '78%' }}>
                        <ProgressIndicator
                            barHeight={30}
                            percentComplete={percent}
                        />
                        <Stack horizontal className="boundary">
                            <StackItem grow={30}>0</StackItem>
                            <StackItem className="right" grow={70}>{maxString}</StackItem>
                        </Stack>
                    </div>
                    <div className="description" style={{ width: '22%' }}>{description}</div>
                </Stack>
                <br />
            </div>
        );
    }
}

export default ProgressBar;