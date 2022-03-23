import { FontWeights, mergeStyleSets, getTheme } from '@fluentui/react';

const theme = getTheme();

export const styles = mergeStyleSets({
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
    ],
    buttons: {
        display: 'flex',
        justifyContent: 'flex-end',
        padding: '0 24px 24px'
    }
});
